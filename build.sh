#!/usr/bin/env bash
# build.sh

# Fail on error
apt-get update
apt-get install -y tesseract-ocr
apt-get install -y tesseract-ocr-eng
apt-get install -y tesseract-ocr-mal
apt-get install -y fonts-lohit-mlym

# Verify Malayalam OCR is installed
if ! tesseract --list-langs 2>/dev/null | grep -Eiq '^mal$'; then
  echo "Downloading Malayalam traineddata..."
  curl -fsSL -o /usr/share/tesseract-ocr/4.00/tessdata/mal.traineddata \
    https://github.com/tesseract-ocr/tessdata_fast/raw/main/mal.traineddata
fi

# Install Python dependencies
pip install -r requirements.txt
