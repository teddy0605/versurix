# versurix

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/teddy0605/versurix.svg?style=social)](https://github.com/teddy0605/versurix)

Local lyrics from YouTube (or audio files): **yt-dlp → mlx-whisper** on Apple Silicon. No cloud or API keys.

## Requirements

- macOS, Apple Silicon  
- Python 3.11+  
- `ffmpeg` — `brew install ffmpeg`

## Install

```bash
git clone https://github.com/teddy0605/versurix
cd versurix
python3 -m venv .venv && source .venv/bin/activate
pip install --editable .
```

For **`--isolate-vocals`**, install Demucs once: **`pip install --editable ".[vocals]"`** (adds PyTorch; large). 
Tests: **`pip install --editable ".[dev]"`**.

First run downloads the Whisper model (~800MB, cached under `~/.cache/huggingface/hub/`).

## Usage

**URLs:** Pass **one or more links as arguments** (space-separated — each token is one URL). If you pass none, URLs are taken from the **`urls`** array in `versurix_config.json`, or from **`VERSURIX_URL`** / **`VERSURIX_URLS`** (comma-separated).

**zsh and `?`:** Quote URLs that contain `?`, or escape **`watch\?v=`**.

```bash
versurix 'https://www.youtube.com/watch?v=VIDEO_ID'
versurix 'https://youtu.be/VIDEO_A' 'https://youtu.be/VIDEO_B'
VERSURIX_URL=https://www.youtube.com/watch?v=VIDEO_ID versurix
```

| Flag | Notes |
|------|--------|
| `--language` | BCP-47 or `auto` (default `es`) |
| `--model` | HF repo (default `mlx-community/whisper-large-v3-turbo`) |
| `--output-format` | `txt`, `srt`, `both` |
| `--output-dir` | Where to write lyrics, SRT, and kept MP3s (default `downloads/`) |
| `--isolate-vocals` | Demucs — needs **`pip install --editable ".[vocals]"`** first |
| `--enhance-vocals` | FFmpeg EQ on vocals |
| `--keep-audio` | Keep MP3 |
| `--local` | Input paths are local audio (skip yt-dlp) |
| `--verbose` | Whisper tqdm on stderr |
| `--config` | Config file (default `./versurix_config.json`) |

## Configuration

Edit **`versurix_config.json`** in your working directory. Keys mirror CLI options in **snake_case** (`isolate_vocals`, `keep_audio`, etc.). Optional **`urls`** is a JSON array of strings when you want to run **`versurix`** with no URL arguments. Optional **`whisper`** merges onto the built-in decode defaults — see the committed example file for the full shape.

If the config file is missing, built-in defaults apply.

## Examples

```bash
versurix 'https://www.youtube.com/watch?v=VIDEO_ID' --language en --output-format srt
versurix 'https://a' 'https://b' --isolate-vocals   # after pip install --editable ".[vocals]"
versurix --local "./track.mp3"
# Or list URLs in versurix_config.json ("urls") / env, then: versurix --config ./versurix_config.json
```

## Output

By default files go under **`downloads/`** (create with `mkdir` on run). Override with **`output_dir`** in config or **`--output-dir`**.

- **`downloads/{title}.txt`**  
- **`downloads/{title}.srt`** when format is `srt` or `both`  
- With **`--keep-audio`**, the MP3 is stored in the same output directory  
- Logs on stdout

## Tips

Heavy instrumentation or long intros: **`--isolate-vocals`** (first run pulls ~80MB Demucs). For more accuracy at the cost of size, try **`--model mlx-community/whisper-large-v3-mlx`**.
