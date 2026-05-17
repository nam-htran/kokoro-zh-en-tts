---
title: Kokoro 82M v1.1 zh TTS
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
python_version: "3.10"
suggested_hardware: cpu-upgrade
pinned: false
license: apache-2.0
---

# Kokoro 82M v1.1 zh TTS

Gradio Space for `hexgrad/Kokoro-82M-v1.1-zh` with the built-in Kokoro voices.

Live demo: https://huggingface.co/spaces/hnamt/kakoro-TTS-zh-en

## Deploy

Create a new Hugging Face Space with the Gradio SDK, then push these files:

```bash
git init
git add .gitignore app.py environment.yml requirements.txt packages.txt README.md
git commit -m "Create Kokoro zh TTS Space"
git branch -M main
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```

The Space downloads model weights and selected voices from Hugging Face on first run.

## Run locally

Use Python 3.10 or 3.11. With conda:

```bash
conda env create -f environment.yml
conda activate kokoro-tts
python app.py
```

On Linux or Hugging Face Spaces, `espeak-ng` is installed from `packages.txt`.
