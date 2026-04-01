# main.py — PDF QA API (Level-4 route; zero hardcoded mappings, Malayalam-aware OCR, section-aware context)
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Env: API_TOKEN (required), MISTRAL_API_KEY (optional), GROQ_API_KEY (optional)

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
import os, tempfile, io, time, re, json, asyncio, unicodedata, difflib
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import requests, httpx
from dotenv import load_dotenv
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- App & Clients ----------------
app = FastAPI()

REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]),
    ),
)
ASYNC_CLIENT = httpx.AsyncClient(
    http2=True,
    timeout=20.0,
    headers={"Accept-Encoding": "gzip, deflate"},
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

@app.on_event("shutdown")
async def shutdown_event():
    await ASYNC_CLIENT.aclose()

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
API_TOKEN       = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN must be set")

# ---------------- Schemas ----------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- Language helpers ----------------
def is_malayalam(s: str) -> bool:
    return any(0x0D00 <= ord(ch) <= 0x0D7F for ch in (s or ""))

def question_lang_label(q: str) -> str:
    return "Malayalam" if is_malayalam(q) else "English"

def enforce_lang_instruction(q: str) -> str:
    return (
        f"IMPORTANT: Answer in the SAME language as THIS question: {question_lang_label(q)}. "
        f"Do NOT switch languages."
    )

# ---------------- Prompts ----------------
FULL_PROMPT_TEMPLATE = """You are an insurance policy expert. Use ONLY the information provided in the context to answer the questions.
Context:
{context}

Questions:
{query}

Instructions:
1. Provide clear and direct answers based ONLY on the context.
2. Do not specify the clause number or clause description.
3. If the answer is "Yes" or "No," include a short explanation.
4. If not found in the context, reply: "Not mentioned in the policy."
5. Give each answer in a single paragraph without numbering.

Answers:"""

CHUNK_PROMPT_TEMPLATE = """You are an insurance policy specialist. Prefer answers from the policy <Context> only.

Decision rule:
1) Search ALL of <Context>. If the answer exists there, answer ONLY from <Context>.
2) If the answer is NOT in <Context>, reply exactly: "Not mentioned in the policy."
3) Answer in the SAME LANGUAGE as the question.
4) {lang_rule}

Requirements:
- Quote every number, amount, time period, percentage, sub-limit, definition, eligibility, exclusion, waiting period, and condition **word-for-word**.
- If any numbers or conditions are present in the answer, include at least one short **verbatim** quote from <Context>.
- If Yes/No, start with “Yes.” or “No.” and immediately quote the rule that makes it so.
- Include all applicable conditions in a compact way.
- No clause numbers, no speculation, no invented facts.

Context:
{context}

Questions:
{query}

Answers (one concise paragraph per question, no bullets, no numbering):
"""

WEB_PROMPT_TEMPLATE = """You are an expert insurance policy assistant. Based on the document titled "{title}", answer the following questions using general or public insurance knowledge.
Title: "{title}"

Questions:
{query}

Instructions:
- Use public knowledge.
- If specific document data is needed, reply: "Not found in public sources."
- Answer EACH question in the SAME LANGUAGE it is asked (English → English, Malayalam → Malayalam). Do not mix languages across answers.
- Keep each answer concise (1 paragraph max).
- Give each answer in a single paragraph without numbering.

Answers:"""

# ---------------- Helpers ----------------
def approx_tokens_from_text(s: str) -> int:
    return max(1, len(s) // 4)

def choose_mistral_params(page_count: int, context_text: Optional[str]):
    # Deterministic for ≤100 pages to reduce hallucination
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 100:
        max_tokens, temperature, timeout = 1100, 0.06, 18
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1300, 0.14, 15
    else:
        max_tokens, temperature, timeout = 700, 0.18, 12
    total_budget = 3600
    budget_left = max(800, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}

def choose_groq_params(page_count: int, context_text: Optional[str]):
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 100:
        max_tokens, temperature, timeout = 1300, 0.0, 30
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1700, 0.18, 30
    else:
        max_tokens, temperature, timeout = 1100, 0.13, 25
    total_budget = 3600
    budget_left = max(800, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}

def make_question_block(questions: List[str]) -> str:
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

# ---------------- OCR Lang decision ----------------
def _pick_ocr_lang(any_malayalam_q: bool, sample_text: str) -> str:
    # If any Q is Malayalam OR sample shows Malayalam, use eng+mal
    if any_malayalam_q or is_malayalam(sample_text):
        return "eng+mal"
    return "eng"

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf_url(pdf_url: str, prefer_lang: str) -> Tuple[str, int, str]:
    r = REQUESTS_SESSION.get(pdf_url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    full_text, title = "", ""
    with fitz.open(tmp_path) as doc:
        page_count = len(doc)

        # Title from first few pages; OCR fallback with prefer_lang
        for i in range(min(15, page_count)):
            t = (doc[i].get_text() or "").strip()
            if not t:
                try:
                    pix = doc[i].get_pixmap(dpi=140)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        t = pytesseract.image_to_string(img, lang=prefer_lang).strip()
                    except Exception:
                        t = pytesseract.image_to_string(img, lang="eng").strip()
                except Exception:
                    continue
            if t:
                title = t.splitlines()[0][:120]
                break

        # Full text (≤200 pages)
        if page_count <= 200:
            for i in range(page_count):
                t = (doc[i].get_text() or "").strip()
                if len(t) < 200:  # OCR when page is image-heavy or too sparse
                    try:
                        pix = doc[i].get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                        if img.width < 1500:
                            img = img.resize((int(img.width * 1.4), int(img.height * 1.4)))
                        img = img.point(lambda p: 255 if p > 180 else 0)
                        t_ocr = pytesseract.image_to_string(img, lang=prefer_lang).strip()
                        if len(t_ocr) > len(t):
                            t = t_ocr
                    except Exception:
                        pass
                if t:
                    full_text += t + "\n"

    os.remove(tmp_path)
    return (full_text.strip() if page_count <= 200 else "", page_count, title or "Untitled Document")

def split_text(text: str, chunk_size: int = 1200, overlap: int = 350) -> List[str]:
    chunks, start = [], 0
    n = len(text)
    while start < n and len(chunks) < 20:
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def _fast_title_and_snippets_with_ocr(data: bytes, prefer_lang: str) -> tuple[int, str, str]:
    page_count = 0
    title = "Untitled Document"
    snippets = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            sample_idxs = [i for i in (0, 1, page_count - 1) if 0 <= i < page_count]
            for idx in sample_idxs:
                page = doc[idx]
                txt = (page.get_text("text", sort=True) or "").strip()
                if not txt:
                    def _ocr_at_dpi(dpi: int) -> str:
                        pix = page.get_pixmap(dpi=dpi)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                        if img.width < 1500:
                            img = img.resize((int(img.width * 1.5), int(img.height * 1.5)))
                        img = img.point(lambda p: 255 if p > 180 else 0)
                        return (pytesseract.image_to_string(img, lang=prefer_lang) or "").strip()
                    txt = _ocr_at_dpi(140)
                    if len(txt) < 400:
                        txt = _ocr_at_dpi(170)
                if idx in (0, 1) and txt and title == "Untitled Document":
                    title = txt.splitlines()[0][:120]
                if txt:
                    snippets.append("\n".join(txt.splitlines()[:30])[:1500])
    except Exception:
        pass
    tiny = ("\n\n---\n\n").join(snippets)[:3000] if snippets else ""
    return page_count, title, tiny




# ---------------- LLM Calls ----------------



# ---------------- LLM Calls ----------------

def call_mistral(prompt: str, params: dict, system: Optional[str] = None) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": "mistral-small-latest",
        "temperature": params.get("temperature", 0.0),
        "top_p": 1,
        "max_tokens": params.get("max_tokens", 512),
        "messages": messages,
    }
    r = REQUESTS_SESSION.post(url, headers=headers, json=payload, timeout=params.get("timeout", 15))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Strict, context-only helper for ticket PDF (used ONLY if parser is unsure)
LANDMARK_PROMPT = """
You are helping map a CITY to its LANDMARK and to an ENDPOINT using ONLY the <TicketContext> text below.
Do NOT use any public or outside knowledge. If the mapping is not present in <TicketContext>, say "CANNOT-DETERMINE".

<TicketContext>
{ticket_text}
</TicketContext>

Rules to follow:
- First, read the "Landmark  Current Location" tables and determine the LANDMARK for the favourite CITY exactly as listed.
- Then, read the "Step 3: Choose Your Flight Path" rules and select the correct endpoint NAME (the last path segment like getThirdCityFlightNumber) for that LANDMARK.
- If the LANDMARK has an explicit rule, use that endpoint. If not, use the "For all other landmarks" endpoint.
- Return a single line in the exact format: LANDMARK=..., ENDPOINT=...
- If you truly cannot determine from <TicketContext>, return exactly: CANNOT-DETERMINE

Favourite CITY: "{city}"
"""

# ---------------- Relevance & Evidence (generic; no domain words) ----------------
TOKEN_RX = re.compile(r"\w+", flags=re.UNICODE)  # Unicode-aware tokens (Malayalam supported)
NUMLIKE_RX = re.compile(
    r"(\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\b|\d+\s*(?:%|days?|months?|years?|weeks?|hrs?|hours?)|\u20B9|\$)",
    re.UNICODE | re.IGNORECASE
)

def _tokens(s: str) -> List[str]:
    return [w.lower() for w in TOKEN_RX.findall(s or "")]

def _score_chunk(q: str, c: str) -> int:
    qt = set(_tokens(q))
    ct = set(_tokens(c))
    base = len(qt & ct)
    num_bonus = min(6, len(NUMLIKE_RX.findall(c)))  # reward numeric/temporal mentions
    return base + num_bonus

def _topk_chunks(q: str, chunks: List[str], k: int = 4) -> List[str]:
    scored = sorted((( _score_chunk(q, c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]] if scored else []

def _harvest_numeric_lines(text: str, max_lines: int = 100) -> str:
    seen, out = set(), []
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        if NUMLIKE_RX.search(ln) and ln not in seen:
            seen.add(ln); out.append(ln)
            if len(out) >= max_lines: break
    return "\n".join(out)

# -------- Section indexer (generic, no hardcoded topic names) --------
HEADING_RX = re.compile(
    r"^(\d+(\.\d+)*\s+.+|[A-Z][A-Z0-9\s\-/,&()]{3,}$|[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+.*)$"
)

def _build_section_index(text: str) -> List[Dict[str, Any]]:
    lines = (text or "").splitlines()
    offsets = []
    pos = 0
    for ln in lines:
        st = ln.strip()
        if HEADING_RX.match(st) and len(st) <= 140:
            offsets.append((pos, st))
        pos += len(ln) + 1
    if not offsets:
        return [{"title": "Document", "start": 0, "end": len(text), "body": text}]
    sections = []
    for i, (start, title) in enumerate(offsets):
        end = offsets[i+1][0] if i+1 < len(offsets) else len(text)
        body = text[start:end]
        sections.append({"title": title, "start": start, "end": end, "body": body})
    return sections

def _section_similarity(q: str, title: str) -> float:
    ql, tl = (q or "").lower(), (title or "").lower()
    sm = difflib.SequenceMatcher(a=ql, b=tl).ratio()
    qset, tset = set(_tokens(q)), set(_tokens(title))
    jacc = len(qset & tset) / max(1, len(qset | tset))
    return 0.6*sm + 0.4*jacc

def _select_relevant_sections(q: str, sections: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    scored = sorted((( _section_similarity(q, s["title"]), s) for s in sections), key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]] if scored else []

def _context_merge_with_sections(q: str, chunks: List[str], sections: List[Dict[str, Any]], budget_chars: int = 9500) -> str:
    buf = []
    used = 0
    # 1) chunk-level relevance with neighbors
    idxs = sorted(((i, _score_chunk(q, c)) for i, c in enumerate(chunks)), key=lambda x: x[1], reverse=True)
    top_idxs = [i for i, _ in idxs[:6]]
    want = set()
    for i in top_idxs:
        for j in range(max(0, i-1), min(len(chunks), i+2)):
            want.add(j)
    for j in sorted(want):
        seg = chunks[j].strip()
        if not seg: continue
        if used + len(seg) + 2 > budget_chars: break
        buf.append(seg); used += len(seg) + 2
    # 2) add best sections by heading similarity
    for sec in _select_relevant_sections(q, sections, k=3):
        seg = (sec.get("body") or "").strip()
        if not seg: continue
        if used + len(seg) + 2 > budget_chars: break
        buf.append(seg); used += len(seg) + 2
    # 3) numeric evidence lines
    evidence = _harvest_numeric_lines("\n\n".join(buf), max_lines=120)
    if evidence:
        buf.append("\n--- Evidence ---\n" + evidence)
    return "\n\n".join(buf)

# ---------------- Level-4 helpers ----------------
_NOT_FOUND_RX = re.compile(r"^\s*not\s+mentioned\s+in\s+the\s+policy\.?\s*$", re.I)

def _sanitize_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\d+[\.\)]\s*", "", s)
    s = re.sub(r"^Answer\s*:\s*", "", s, flags=re.I)
    return s.strip()

def _is_not_found(s: str) -> bool:
    return bool(_NOT_FOUND_RX.match((s or "").strip()))

def _single_paragraph(s: str) -> str:
    s = s.replace("\r", " ")
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

_MAL = r"\u0D00-\u0D7F"

def _fix_malayalam_spacing(s: str) -> str:
    s = re.sub(rf"([{_MAL}])(\d)", r"\1 \2", s)
    s = re.sub(rf"(\d)([{_MAL}])", r"\1 \2", s)
    s = re.sub(r"(\d)\s*%", r"\1 %", s)
    s = re.sub(rf"([{_MAL}])\s*(\d+)\s*-\s*([{_MAL}])", r"\1 \2-\3", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _postshape(a: str, target_lang: str) -> str:
    a = (a or "").strip()
    # strip meta lines and reflow
    a = re.sub(r"^\s*(no direct quote|related verbatim quote|note|explanation)\s*:.*$", "", a, flags=re.I | re.M)
    a = _single_paragraph(a)
    a = a.rstrip(" .")
    if target_lang == "Malayalam":
        a = _fix_malayalam_spacing(a)
    return a

# --- Retry (language-enforced) ---
async def _retry_per_question(q: str, chunks: List[str], page_count: int, sections: Optional[List[Dict[str, Any]]] = None) -> str:
    lang_rule = enforce_lang_instruction(q)
    sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."
    try:
        ctx = _context_merge_with_sections(q, chunks, sections or [], budget_chars=9500)
        m_params = choose_mistral_params(page_count, ctx)
        a = _sanitize_line(call_mistral(
            CHUNK_PROMPT_TEMPLATE.format(context=ctx, query=q, lang_rule=lang_rule),
            m_params,
            system=sys_msg
        ))
        if a and not _is_not_found(a):
            return _postshape(a, question_lang_label(q))
    except Exception:
        pass
    try:
        g_params = choose_groq_params(page_count, "\n".join(chunks[:10]))
        ctx2 = _context_merge_with_sections(q, chunks, sections or [], budget_chars=9000)
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": g_params.get("temperature", 0.0),
            "top_p": 1,
            "max_tokens": g_params.get("max_tokens", 1000),
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": CHUNK_PROMPT_TEMPLATE.format(context=ctx2, query=q, lang_rule=lang_rule)}
            ],
        }
        r = await ASYNC_CLIENT.post(url, headers=headers, json=payload, timeout=g_params.get("timeout", 20))
        r.raise_for_status()
        a2 = _sanitize_line(r.json()["choices"][0]["message"]["content"].strip())
        if a2:
            return _postshape(a2, question_lang_label(q))
    except Exception:
        pass
    return "Not mentioned in the policy."

# --- detect PDF vs mission URL ---
def _is_pdf_payload(data: bytes, ctype: str) -> bool:
    return (ctype and "pdf" in ctype.lower()) or data.startswith(b"%PDF")

def _is_mission_host(url: str) -> bool:
    try:
        return "register.hackrx.in" in urlparse(url).netloc.lower()
    except Exception:
        return False

# --- token extraction (generic, handles plaintext token bodies) ---
TOKEN_KEYS = ("secret_token", "secretToken", "token", "apiKey", "apikey", "key", "secret")

def _extract_token_from_json(js):
    if isinstance(js, dict):
        for k, v in js.items():
            if k in TOKEN_KEYS and isinstance(v, (str, int)):
                return str(v)
        for v in js.values():
            t = _extract_token_from_json(v)
            if t:
                return t
    elif isinstance(js, list):
        for it in js:
            t = _extract_token_from_json(it)
            if t:
                return t
    return None

def _handle_mission_url(data: bytes) -> Optional[str]:
    raw = data.decode("utf-8", errors="ignore").strip().strip('"').strip()
    # 1) JSON hunt
    try:
        js = json.loads(raw)
        t = _extract_token_from_json(js)
        if t:
            return t
    except Exception:
        pass
    # 2) Plaintext: if whole body looks like a token, just return it
    if re.fullmatch(r"[A-Za-z0-9._\-]{16,}", raw):
        return raw
    # 3) Fallback: search inside text
    m = re.search(r'(?:(?:secret\s*token|secretToken|token|api[_\s-]*key|apikey|key)\s*[:=]\s*["\']?)([A-Za-z0-9._\-]{16,})', raw, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r'<(?:code|pre)[^>]*>\s*([^<>\s]{16,})\s*</(?:code|pre)>', raw, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,128}(?![A-Za-z0-9])', raw)
    if m:
        return m.group(0)
    m = re.search(r'(?<![A-Za-z0-9])[A-Za-z0-9._\-]{24,256}(?![A-Za-z0-9])', raw)
    if m:
        return m.group(0)
    return None

# ---- Ticket PDF parsing (no hardcoded city/landmark list) ----

def _parse_city_landmark_pairs(ticket_text: str) -> Dict[str, str]:
    """
    Parse the two 'Landmark  Current Location' tables dynamically (no hardcoding).
    Returns { city -> landmark } and is robust to multi-word cities like 'New York' or 'Dubai Airport'.
    """
    city_to_landmark: Dict[str, str] = {}
    lines = [ln.strip() for ln in (ticket_text or "").splitlines()]

    def is_block_break(ln: str) -> bool:
        low = (ln or "").lower()
        return (
            not ln
            or low.startswith("page ")
            or low.startswith("mission ")
            or low.startswith("step ")
            or "final deliverable" in low
        )

    def looks_titlecase_block(s: str) -> bool:
        toks = [t for t in s.split() if t]
        return len(toks) > 0 and all(t[0].isupper() or t.isupper() for t in toks)

    def split_landmark_city(ln: str) -> Optional[Tuple[str, str]]:
        """
        Choose best split between landmark (left) and city (right) by scoring k in {1,2,3}.
        Prefers 1–3 token cities and 2+ token landmarks; allows multi-word cities like 'New York', 'Los Angeles', 'Cape Town', 'Dubai Airport'.
        """
        toks = [t for t in ln.split() if t]
        if len(toks) < 2:
            return None

        best = None
        best_score = -10
        for k in (1, 2, 3):
            if len(toks) - k < 1:
                continue
            left = " ".join(toks[: len(toks) - k]).strip()
            right = " ".join(toks[len(toks) - k :]).strip()
            if not left or not right:
                continue

            l_toks = left.split()
            r_toks = right.split()

            score = 0
            if looks_titlecase_block(right):
                score += 3
            if 1 <= len(r_toks) <= 3:
                score += 2
            if len(l_toks) >= 2:
                score += 2
            if re.search(r"\b(of|the|and|de|la)\b$", left, flags=re.I):
                score -= 1
            if len(r_toks) == 1 and r_toks[0].lower() in {"airport"}:
                score -= 2

            if score > best_score:
                best_score = score
                best = (left, right)

        if best:
            return best
        return (" ".join(toks[:-1]).strip(), toks[-1].strip())

    parsing = False
    for ln in lines:
        low = ln.lower()
        if "landmark current location" in low:
            parsing = True
            continue
        if not parsing:
            continue
        if is_block_break(ln):
            parsing = False
            continue
        # Skip section headers but keep parsing across both tables
        if low in {"indian cities", "international cities"}:
            continue
        if not ln:
            continue

        pair = split_landmark_city(ln)
        if not pair:
            continue
        landmark, city = pair
        if landmark.lower() == "landmark" or city.lower() == "current location":
            continue
        city_to_landmark[city.strip()] = landmark.strip()

    return city_to_landmark

def _parse_landmark_to_endpoint(ticket_text: str) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Parse 'Step 3: Choose Your Flight Path' rules into:
      - lm_to_ep: { landmark -> endpoint_name }
      - default_ep: endpoint_name for 'For all other landmarks'
    """
    lm_to_ep: Dict[str, str] = {}
    default_ep: Optional[str] = None
    txt = ticket_text or ""

    rule_rx = re.compile(
        r'If\s+landmark.*?(?:is|=)\s*[“"”]([^"”]+)[”"]\s*,?\s*call\s*:\s*GET\s+(https?://\S+)',
        flags=re.I | re.S
    )
    for m in rule_rx.finditer(txt):
        landmark = (m.group(1) or "").strip()
        ep_url = (m.group(2) or "").strip()
        ep_name = ep_url.rstrip("/").split("/")[-1] if ep_url else ""
        if landmark and ep_name:
            lm_to_ep[landmark] = ep_name

    def_rx = re.compile(r'For\s+all\s+other\s+landmarks.*?GET\s+(https?://\S+)', flags=re.I | re.S)
    m = def_rx.search(txt)
    if m:
        ep_url = (m.group(1) or "").strip()
        default_ep = ep_url.rstrip("/").split("/")[-1] if ep_url else None

    return lm_to_ep, default_ep

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

def _choose_endpoint_for_city(city: str, city_to_landmark: Dict[str, str], lm_to_ep: Dict[str, str], default_ep: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (endpoint_name, matched_landmark) for the given city using ONLY parsed PDF structures.
    """
    if not city:
        return None, None
    for c, lm in city_to_landmark.items():
        if _norm(c) == _norm(city):
            return (lm_to_ep.get(lm) or default_ep, lm)
    for c, lm in city_to_landmark.items():
        if _norm(city) in _norm(c) or _norm(c) in _norm(city):
            return (lm_to_ep.get(lm) or default_ep, lm)
    return default_ep, None

def _get_favourite_city() -> str:
    city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
    r1 = REQUESTS_SESSION.get(city_url, timeout=12, headers={"Accept": "application/json"})
    r1.raise_for_status()
    try:
        j = r1.json() or {}
        data = j.get("data") if isinstance(j.get("data"), dict) else j
        city = (data or {}).get("city", "")
        if city:
            return str(city).strip()
    except Exception:
        pass
    return (r1.text or "").strip().strip('"').strip()

def _call_flight_endpoint(ep_name: str) -> str:
    url = f"https://register.hackrx.in/teams/public/flights/{ep_name}"
    print(f"[Ticket] Calling endpoint: {url}")
    r = REQUESTS_SESSION.get(url, timeout=12, headers={"Accept": "application/json"})
    r.raise_for_status()
    # Try JSON
    try:
        j = r.json() or {}
        data = j.get("data") if isinstance(j.get("data"), dict) else j
        flight = (data or {}).get("flightNumber") or (data or {}).get("flight_number") or (data or {}).get("flight")
        if flight:
            print(f"[Ticket] Flight number (JSON): {flight}")
            return str(flight).strip()
    except Exception:
        pass
    # Fallback: regex or raw text
    txt = (r.text or "").strip()
    m = re.search(r'"?flight[_ ]?number"?\s*[:=]\s*"?([A-Za-z0-9]+)"?', txt, flags=re.I)
    if m:
        flight = m.group(1)
        print(f"[Ticket] Flight number (regex): {flight}")
        return flight
    print(f"[Ticket] Flight raw body: {txt[:200]}")
    return txt.strip('"').strip()

def _solve_flight_number_via_ticket_pdf(ticket_pdf_url: str) -> str:
    """
    Parse mapping + rules from the given ticket PDF URL (no hardcoded tables),
    fetch favourite city, choose endpoint, return flight number.
    """
    # Grab PDF & text
    rt = REQUESTS_SESSION.get(ticket_pdf_url, timeout=20)
    rt.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(rt.content)
        path = tmp.name
    text = ""
    with fitz.open(path) as doc:
        for i in range(len(doc)):
            t = (doc[i].get_text() or "").strip()
            if len(t) < 200:
                try:
                    pix = doc[i].get_pixmap(dpi=140)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                    if img.width < 1500:
                        img = img.resize((int(img.width * 1.3), int(img.height * 1.3)))
                    img = img.point(lambda p: 255 if p > 180 else 0)
                    t_ocr = pytesseract.image_to_string(img, lang="eng").strip()
                    if len(t_ocr) > len(t):
                        t = t_ocr
                except Exception:
                    pass
            if t:
                text += t + "\n"
    try:
        os.remove(path)
    except Exception:
        pass

    text_norm = unicodedata.normalize("NFKC", text or "")
    city_to_landmark = _parse_city_landmark_pairs(text_norm)
    lm_to_ep, default_ep = _parse_landmark_to_endpoint(text_norm)

    # Debug: show first few pairs and rules
    print(f"[Ticket] Parsed {len(city_to_landmark)} city→landmark pairs.")
    if city_to_landmark:
        sample_items = list(city_to_landmark.items())[:6]
        print(f"[Ticket] Sample pairs: {sample_items}")
    print(f"[Ticket] Landmark rules: {lm_to_ep} | default={default_ep}")

    fav_city = _get_favourite_city()
    print(f"[Ticket] Favourite city: {fav_city}")

    ep_name, matched_landmark = _choose_endpoint_for_city(fav_city, city_to_landmark, lm_to_ep, default_ep)

    # If we failed to map, consult the LLM ON THE PDF TEXT ONLY (optional)
    if (not ep_name or not matched_landmark) and MISTRAL_API_KEY:
        try:
            llm_out = call_mistral(
                LANDMARK_PROMPT.format(ticket_text=text_norm, city=fav_city),
                {"max_tokens": 128, "temperature": 0.0, "timeout": 12},
                system="Use ONLY the provided TicketContext. Never use public knowledge."
            ).strip()
            print(f"[Ticket][LLM] Raw mapping output: {llm_out}")
            if llm_out and llm_out != "CANNOT-DETERMINE":
                lm_m = re.search(r"LANDMARK\s*=\s*(.+?),\s*ENDPOINT\s*=\s*([A-Za-z0-9]+)", llm_out, flags=re.I)
                if lm_m:
                    matched_landmark = lm_m.group(1).strip()
                    ep_name = lm_m.group(2).strip()
        except Exception as _:
            pass

    print(f"[Ticket] Matched landmark: {matched_landmark}")
    print(f"[Ticket] Chosen endpoint: {ep_name}")

    if not ep_name:
        raise RuntimeError("Could not choose endpoint from ticket PDF rules")

    flight = _call_flight_endpoint(ep_name)
    print(f"[Ticket] Final flight number: {flight}")
    return flight

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return {"message": "Bajaj Chatbot PDF API is running"}

@app.post("/api/v1/hackrx/run")
async def run_analysis_final(request: RunRequest, authorization: str = Header(...)):
    """
    Final Level-4 route (no hardcoded mappings):
    - Token/secret endpoints: returns exact token (handles plaintext).
    - Ticket/flight: parses city↔landmark table & route rules from the provided ticket PDF; picks endpoint; returns flight number.
    - >200 pages: tiny OCR snippets + title → WEB prompt (early return; no full extraction).
    - ≤100 pages: Malayalam-aware OCR, section-aware context, numeric-evidence emphasis, guard rewrite.
    """
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    qtext = " ".join(request.questions).lower()
    start = time.time()

    try:
        # Always fetch the provided documents once (could be PDF, token URL, or anything)
        r = REQUESTS_SESSION.get(request.documents, timeout=20)
        r.raise_for_status()
        data = r.content
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].lower().strip()
        is_pdf = _is_pdf_payload(data, ctype)

        # ---- TOKEN / SECRET HANDLING (plaintext or JSON) ----
        if any(x in qtext for x in ("token", "secret", "api key", "apikey", "key")) and not is_pdf:
            token = _handle_mission_url(data)
            return {"answers": [token] if token else ["Not found in non-PDF URL."]}

        # ---- FLIGHT / TICKET NUMBER ----
        if re.search(r"\b(flight|ticket)\s*(no\.?|number)\b", qtext):
            try:
                if is_pdf:
                    flight = _solve_flight_number_via_ticket_pdf(request.documents)
                    return {"answers": [flight]}
                else:
                    # Without a ticket PDF context, last-resort probe of known endpoints (still no public info)
                    for ep in ("getFirstCityFlightNumber","getSecondCityFlightNumber","getThirdCityFlightNumber","getFourthCityFlightNumber","getFifthCityFlightNumber"):
                        try:
                            flight = _call_flight_endpoint(ep)
                            if flight and len(flight) >= 3:
                                return {"answers": [flight]}
                        except Exception:
                            continue
                    raise RuntimeError("Flight number not found")
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Flight solver failed: {e}")

        # ---------- For PDFs: decide OCR language ----------
        page_count_fast, title_fast, tiny_snippets = _fast_title_and_snippets_with_ocr(data, prefer_lang="eng")
        prefer_lang = _pick_ocr_lang(any(is_malayalam(q) for q in request.questions), (title_fast or "") + " " + (tiny_snippets or ""))

        # >200 pages → hybrid path: tiny OCR context + public knowledge (no full extraction)
        if page_count_fast and page_count_fast > 200:
            try:
                qblock = make_question_block(request.questions)
                hybrid_title = (title_fast or "Untitled Document")
                if tiny_snippets:
                    hybrid_title = f"{hybrid_title}\n\nKey OCR snippets:\n{tiny_snippets}"
                m_params = choose_mistral_params(page_count_fast, hybrid_title)
                m_params["temperature"] = min(m_params.get("temperature", 0.14), 0.12)
                resp = call_mistral(
                    WEB_PROMPT_TEMPLATE.format(title=hybrid_title, query=qblock),
                    m_params
                )
                cleaned = [_sanitize_line(ln) for ln in resp.splitlines() if ln.strip()]
                answers = cleaned[:len(request.questions)] if cleaned else ["Not found in public sources."] * len(request.questions)
                return {"answers": answers}
            except Exception:
                return {"answers": ["Not found in public sources."] * len(request.questions)}

        # ≤200: Extract full text (Malayalam-aware OCR when needed)
        if is_pdf:
            full_text, page_count, _title = extract_text_from_pdf_url(request.documents, prefer_lang)
            s = unicodedata.normalize("NFKC", full_text or "")
            s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
            s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t ")
            s = re.sub(r"-\s*\n\s*", "", s)
            s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
            full_text = s

            if not full_text:
                raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

            chunks = split_text(full_text) if full_text else []
            sections = _build_section_index(full_text)

            if page_count <= 100:
                answers: List[str] = []
                for q in request.questions:
                    try:
                        ctx = _context_merge_with_sections(q, chunks, sections, budget_chars=9500)
                        m_params = choose_mistral_params(page_count, ctx)
                        sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."
                        a = _sanitize_line(call_mistral(
                            CHUNK_PROMPT_TEMPLATE.format(context=ctx, query=q, lang_rule=enforce_lang_instruction(q)),
                            m_params,
                            system=sys_msg
                        ))
                        if not a or _is_not_found(a):
                            a = await _retry_per_question(q, chunks, page_count, sections=sections)
                    except Exception:
                        a = await _retry_per_question(q, chunks, page_count, sections=sections)

                    # Language guard rewrite (if model slips)
                    try:
                        tgt = question_lang_label(q)
                        if tgt == "Malayalam" and not is_malayalam(a):
                            a = _sanitize_line(call_mistral(
                                f"Rewrite the following answer in Malayalam without adding new facts:\n\n{a}",
                                {"max_tokens": 300, "temperature": 0.0, "timeout": 10},
                                system="Output only the rewritten answer text."
                            ))
                        elif tgt == "English" and is_malayalam(a):
                            a = _sanitize_line(call_mistral(
                                f"Rewrite the following answer in English without adding new facts:\n\n{a}",
                                {"max_tokens": 300, "temperature": 0.0, "timeout": 10},
                                system="Output only the rewritten answer text."
                            ))
                    except Exception:
                        pass

                    answers.append(_postshape(a, question_lang_label(q)))
                return {"answers": answers[:len(request.questions)]}

            # 101–200 pages
            out: List[str] = []
            for q in request.questions:
                try:
                    ctx = _context_merge_with_sections(q, chunks, sections, budget_chars=9500)
                    m_params = choose_mistral_params(page_count, ctx)
                    sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."
                    a = _sanitize_line(call_mistral(
                        CHUNK_PROMPT_TEMPLATE.format(context=ctx, web_snippets="", query=q, lang_rule=enforce_lang_instruction(q)),
                        m_params,
                        system=sys_msg
                    ))
                    if not a or _is_not_found(a):
                        a = await _retry_per_question(q, chunks, page_count, sections=sections)
                    out.append(_postshape(a, question_lang_label(q)))
                except Exception:
                    tmp = await _retry_per_question(q, chunks, page_count, sections=sections)
                    out.append(_postshape(tmp, question_lang_label(q)))
            return {"answers": out}

        # Non-PDF, non-mission & not token/flight → nothing to answer from web
        return {"answers": ["Not found in public sources."] * len(request.questions)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        print(f"⏱ Total time: {round(time.time() - start, 2)}s")


