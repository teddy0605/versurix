#!/usr/bin/env python3
"""
versurix — download a YouTube song and extract lyrics using local AI.

Pipeline: yt-dlp (or local file) → mlx-whisper → lyrics file
"""

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOCAL_AUDIO_SUFFIXES: frozenset[str] = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma", ".aiff",
})


# Constants
WHISPER_DEFAULT_MODEL: str = "mlx-community/whisper-large-v3-turbo"

DEFAULT_CONFIG_PATH: Path = Path("versurix_config.json")
# Lyrics, SRT, and kept MP3s (with --keep-audio) go here unless --output-dir / config overrides.
DEFAULT_OUTPUT_DIR: str = "downloads"

# Defaults used when no config file / JSON key is present (see README "Built-in defaults")
BUILTIN_DEFAULTS: Dict[str, Any] = {
    "language": "es",
    "model": WHISPER_DEFAULT_MODEL,
    "output_format": "txt",
    "output_dir": DEFAULT_OUTPUT_DIR,
}

# mlx-whisper decode options tuned for music; override via config "whisper" object
WHISPER_DECODE_DEFAULTS: Dict[str, Any] = {
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.1,
    "logprob_threshold": -2.0,
    "compression_ratio_threshold": 3.0,
}

# Also accepted by mlx_whisper.transcribe(...) when set in config "whisper"
WHISPER_OPTIONAL_KEYS: frozenset[str] = frozenset({
    "temperature",
    "initial_prompt",
    "word_timestamps",
    "prepend_punctuations",
    "append_punctuations",
    "clip_timestamps",
    "hallucination_silence_threshold",
})

WHISPER_ALLOWED_KEYS: frozenset[str] = frozenset(WHISPER_DECODE_DEFAULTS.keys()) | WHISPER_OPTIONAL_KEYS


def merge_whisper_decode(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config['whisper'] onto WHISPER_DECODE_DEFAULTS; pass through allowed optional mlx keys."""
    out: Dict[str, Any] = {**WHISPER_DECODE_DEFAULTS}
    whisper_cfg = config.get("whisper")
    if not isinstance(whisper_cfg, dict):
        return out
    for key, value in whisper_cfg.items():
        if key not in WHISPER_ALLOWED_KEYS:
            logger.warning("Ignoring unknown whisper config key %r (not passed to mlx_whisper)", key)
            continue
        if key == "temperature" and isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out

# Configure logging (stdout; tqdm uses stderr so bars do not interleave)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def quiet_third_party_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _progress_bar_width() -> int:
    """Single width for every tqdm bar (download, Demucs, Whisper)."""
    try:
        c = shutil.get_terminal_size(fallback=(100, 20)).columns
        return max(60, min(c - 2, 120))
    except Exception:
        return 100


# Blank tqdm label column; phase names come from log_section only (avoids “Download” twice, etc.).
_TQDM_DESC_WIDTH = 12


_tqdm_style_installed = False


def apply_progress_ui_style() -> None:
    """Uniform tqdm width; rounded Demucs totals; no duplicate phase words in the bar."""
    global _tqdm_style_installed
    if _tqdm_style_installed:
        return
    import tqdm.std

    _orig_init = tqdm.std.tqdm.__init__

    _bar_default = (
        "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    _bar_seconds = (
        "{desc}{percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} s "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        kwargs["ncols"] = _progress_bar_width()
        kwargs["dynamic_ncols"] = False
        kwargs.setdefault("leave", True)
        kwargs["desc"] = " " * _TQDM_DESC_WIDTH
        unit = kwargs.get("unit")
        if unit == "seconds":
            kwargs["bar_format"] = _bar_seconds
        else:
            kwargs.setdefault("bar_format", _bar_default)
        _orig_init(self, *args, **kwargs)

    tqdm.std.tqdm.__init__ = _patched_init  # type: ignore[method-assign]
    _tqdm_style_installed = True

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*TorchCodec.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*bits_per_sample.*",
    )


def log_section(title: str) -> None:
    """Blank line + titled rule (stdout; tqdm stays on stderr)."""
    logger.info("")
    w = min(52, max(36, _progress_bar_width()))
    logger.info(title)
    logger.info("-" * w)


def log_detail(msg: str) -> None:
    logger.info(f"  {msg}")


def sanitize_output_stem(name: str, max_len: int = 200) -> str:
    """Keep Unicode like the source audio filename; only strip path-unsafe chars."""
    n = unicodedata.normalize("NFC", name).strip()
    out: List[str] = []
    for ch in n:
        if ch in "/\0":
            out.append(" ")
        elif ch == ":":
            out.append(" - ")
        else:
            out.append(ch)
    s = "".join(out).strip(" .")
    if not s:
        return "versurix_track"
    return s[:max_len]


def output_stem(title: str, video_id: str, audio_path: Path) -> str:
    """Prefer the on-disk audio filename stem (matches --keep-audio MP3), else title / id."""
    if audio_path.name and audio_path.exists():
        stem = audio_path.stem.strip()
        if stem:
            return sanitize_output_stem(stem)
    t = sanitize_output_stem(title) if title.strip() else ""
    if t and t != "versurix_track":
        return t
    return sanitize_output_stem(video_id) if video_id.strip() else "versurix_track"


def hub_model_snapshots_ready(model_repo: str) -> bool:
    """True if HuggingFace hub has a non-empty snapshot for this repo (local cache)."""
    key = "models--" + model_repo.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / key / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir():
            try:
                if any(snap.iterdir()):
                    return True
            except OSError:
                continue
    return False


def urls_from_config(config: Dict[str, Any]) -> List[str]:
    raw = config.get("urls")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and not s.startswith("#"):
            out.append(s.replace("\\", ""))
    return out


def urls_from_env() -> List[str]:
    """
    URLs from the environment (no shell globbing on assignment in zsh/bash).

    VERSURIX_URL — one URL. VERSURIX_URLS — comma-separated list (optional second form).
    """
    out: List[str] = []
    for key, delimiter in (
        ("VERSURIX_URL", None),
        ("VERSURIX_URLS", ","),
    ):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        parts = [raw] if delimiter is None else [p.strip() for p in raw.split(delimiter)]
        for p in parts:
            s = p.replace("\\", "").strip()
            if s:
                out.append(s)
    return out


def resolve_urls(positional: List[str], config: Dict[str, Any]) -> List[str]:
    """
    One or more URLs as CLI arguments (space-separated in the shell).

    Order: positional CLI args, else VERSURIX_URL / VERSURIX_URLS, else config[\"urls\"].
    """
    cli: List[str] = []
    for u in positional:
        s = u.replace("\\", "").strip()
        if s:
            cli.append(s)
    if cli:
        return cli
    env_urls = urls_from_env()
    if env_urls:
        return env_urls
    return urls_from_config(config)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="versurix",
        description="Download a YouTube song and extract lyrics using local AI.",
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help=(
            "One or more URLs (space-separated). If omitted: VERSURIX_URL / VERSURIX_URLS (env), "
            "then config \"urls\". In zsh, quote URLs that contain ? or use "
            "VERSURIX_URL=https://youtube.com/watch?v=ID versurix"
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="BCP-47 language code for Whisper, or 'auto' to detect (default: es)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="REPO",
        help=(
            f"HuggingFace repo for mlx-whisper (default: {WHISPER_DEFAULT_MODEL}). "
            "For better recall on songs with heavy instrumentation or instrumental intros, "
            "try mlx-community/whisper-large-v3-mlx (slower, ~1.5GB)."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "srt", "both"],
        default=None,
        help="Output format: txt (lyrics), srt (with timestamps), or both (default: txt)",
    )
    parser.add_argument(
        "--enhance-vocals",
        action="store_true",
        help=(
            "Boost vocal frequencies via FFmpeg EQ before transcribing "
            "(highpass 200Hz + lowpass 4kHz + dynamic normalization). "
            "Fast, no extra model. Helps with heavy instrumentation."
        ),
    )
    parser.add_argument(
        "--isolate-vocals",
        action="store_true",
        help=(
            "Separate vocals from instruments using Demucs before transcribing. "
            "Requires: pip install --editable \".[vocals]\". First run downloads ~80MB model."
        ),
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep downloaded MP3 file after transcribing",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Treat inputs as paths to local audio files (skip yt-dlp). "
            "Use after --keep-audio to re-transcribe the saved MP3."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=f'Directory for lyrics, SRT, and kept MP3s (default: "{DEFAULT_OUTPUT_DIR}/")',
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        metavar="FILE",
        help=f"JSON settings + optional \"urls\" array (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Whisper decoding progress",
    )
    return parser.parse_args()


def load_local_audio(path_str: str) -> Tuple[Path, Dict[str, Any]]:
    """Resolve a local audio path and return (path, minimal info dict for metadata)."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_file():
        logger.error(f"Local audio not found: {path_str}")
        sys.exit(1)
    suf = p.suffix.lower()
    if suf not in LOCAL_AUDIO_SUFFIXES:
        logger.error(
            f"Unsupported audio type {p.suffix!r} for {p}. "
            f"Supported: {', '.join(sorted(LOCAL_AUDIO_SUFFIXES))}"
        )
        sys.exit(1)
    info: Dict[str, Any] = {
        "title": p.stem,
        "id": p.stem,
        "uploader": "local",
        "duration": 0,
    }
    return p, info


def download_audio(url: str, output_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Download audio from URL as MP3.
    
    Args:
        url: YouTube URL or any yt-dlp supported URL
        output_dir: Directory to save the downloaded audio
        
    Returns:
        Tuple containing (path to downloaded MP3, info dictionary)
        
    Raises:
        SystemExit: If download fails or audio file is not found
    """
    try:
        import tqdm
    except ImportError:
        logger.error("Missing dependency: pip install tqdm")
        sys.exit(1)
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError:
        logger.error("Missing dependency: pip install yt-dlp (or: brew install yt-dlp)")
        sys.exit(1)

    pbar: Optional[Any] = None

    def progress_hook(d: Dict[str, Any]) -> None:
        nonlocal pbar
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            n = int(d.get("downloaded_bytes") or 0)
            if pbar is None:
                pbar = tqdm.tqdm(
                    total=int(total) if total else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    file=sys.stderr,
                    mininterval=0.25,
                )
            if total and (pbar.total is None or pbar.total != int(total)):
                pbar.total = int(total)
            upper = pbar.total if pbar.total is not None else n
            pbar.n = min(n, upper)
            pbar.refresh()
        elif d["status"] == "finished" and pbar is not None:
            pbar.close()
            pbar = None

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }],
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [progress_hook],
    }

    info: Dict[str, Any]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    try:
        mp3_path = Path(info["requested_downloads"][0]["filepath"])
    except (KeyError, IndexError):
        mp3_path = output_dir / f"{info.get('title', info['id'])}.mp3"

    if not mp3_path.exists():
        logger.error(f"Expected audio file not found: {mp3_path}")
        sys.exit(1)

    return mp3_path, info


def enhance_vocals_ffmpeg(audio_path: Path, output_dir: Path) -> Path:
    """
    Boost vocal frequency range using FFmpeg EQ.
    Reduces low-end instrumentation to help Whisper detect sung vocals.
    """
    output_path = output_dir / f"{audio_path.stem}_enhanced.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-af", "highpass=f=200,lowpass=f=4000,dynaudnorm",
        "-ar", "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        logger.error(f"FFmpeg enhancement failed: {result.stderr[-300:]}")
        sys.exit(1)
    return output_path


def isolate_vocals_demucs(audio_path: Path, output_dir: Path) -> Path:
    """
    Isolate vocals using Demucs (Meta's music source separator).
    Produces a clean vocals-only WAV — dramatically improves Whisper accuracy
    on songs with heavy instrumentation or long instrumental intros.

    Requires: pip install --editable ".[vocals]"
    First Demucs run downloads ~80MB htdemucs model from HuggingFace.
    """
    try:
        import demucs.separate as demucs_separate
    except ImportError as e:
        logger.error(
            f"Demucs import failed for the Python running versurix:\n  {sys.executable}\n"
            f"Reason: {e}\n"
        )
        sys.exit(1)

    demucs_out = output_dir / "demucs"
    argv = [
        "--two-stems",
        "vocals",
        "-o",
        str(demucs_out),
        str(audio_path),
    ]
    # In-process: same tqdm styling as download/Whisper; stdout chatter discarded.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            demucs_separate.main(argv)
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            logger.error(f"Demucs failed (exit {code!r}).")
            sys.exit(code if isinstance(code, int) else 1)

    vocals_path = demucs_out / "htdemucs" / audio_path.stem / "vocals.wav"
    if not vocals_path.exists():
        # Demucs may use a different subfolder name
        found = list(demucs_out.rglob("vocals.wav"))
        if not found:
            logger.error(f"Demucs vocals output not found under {demucs_out}")
            sys.exit(1)
        vocals_path = found[0]

    return vocals_path


def transcribe_audio(
    audio_path: Path,
    language: str,
    model_repo: str,
    verbose: bool,
    whisper_decode: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Transcribe audio with mlx-whisper.
    
    Args:
        audio_path: Path to audio file
        language: BCP-47 language code or 'auto'
        model_repo: HuggingFace model repository
        verbose: Whether to show transcription progress
        whisper_decode: mlx-whisper decode kwargs (from config whisper + defaults)
        
    Returns:
        Dictionary containing transcription results with 'text' and 'segments'
        
    Raises:
        SystemExit: If transcription fails
    """
    lang: Optional[str] = None if language == "auto" else language

    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if hub_model_snapshots_ready(model_repo):
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        import mlx_whisper
    except ImportError:
        logger.error("Error: mlx-whisper not installed. Run: pip install mlx-whisper")
        logger.error("Note: mlx-whisper requires Apple Silicon (M1/M2/M3) and macOS")
        sys.exit(1)

    def run_mlx_transcribe() -> Dict[str, Any]:
        # tqdm>=4.67 has no tqdm.disable; mlx uses per-bar disable via verbose=
        return mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_repo,
            language=lang,
            verbose=verbose,
            **whisper_decode,
        )

    try:
        result = run_mlx_transcribe()
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        logger.error("This might happen if the download was interrupted or failed")
        sys.exit(1)
    except Exception as e:
        msg = str(e).lower()
        hub_like = any(
            x in msg
            for x in (
                "offline",
                "connect",
                "connection",
                "network",
                "hub",
                "huggingface",
                "couldn't",
                "could not",
                "resolve",
                "timeout",
                "401",
                "403",
                "404",
                "repository",
                "revision",
            )
        )
        if os.environ.get("HF_HUB_OFFLINE") == "1" and hub_like:
            os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                logger.warning("HF offline mode failed; retrying with hub online (incomplete cache?)…")
                result = run_mlx_transcribe()
                return result
            except Exception as e2:
                logger.error(f"Transcription failed: {e2}")
                sys.exit(1)
        logger.error(f"Transcription failed: {e}")
        logger.error("Common issues: insufficient disk space, model download interrupted")
        sys.exit(1)

    return result


def _format_srt_time(seconds: float) -> str:
    """
    Format seconds as SRT timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted SRT timestamp string (HH:MM:SS,mmm)
    """
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert Whisper segments to SRT format string.
    
    Args:
        segments: List of segment dictionaries from Whisper
        
    Returns:
        SRT formatted string
    """
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """
    Fill in args that were not explicitly set on the CLI using config values,
    falling back to hardcoded defaults. Uses sentinel None to detect unset args.
    """
    for key, hardcoded_default in BUILTIN_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, hardcoded_default))

    args.output_dir = Path(args.output_dir)

    for flag in ("keep_audio", "verbose", "enhance_vocals", "isolate_vocals", "local"):
        if not getattr(args, flag) and config.get(flag):
            setattr(args, flag, True)

    args.whisper = merge_whisper_decode(config)

    return args


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file, or None
        
    Returns:
        Dictionary of configuration settings
    """
    if not config_path or not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw: Any = json.load(f)
        if isinstance(raw, dict):
            return raw
        return {}
    except Exception as e:
        logger.warning(f"Could not load config file {config_path}: {e}")
        return {}


def main() -> None:
    """
    Main entry point for versurix.
    
    Handles the complete pipeline:
    1. Parse arguments
    2. Download audio
    3. Transcribe with Whisper
    4. Save output files
    """
    apply_progress_ui_style()
    quiet_third_party_loggers()
    args = parse_args()

    config = load_config(args.config)
    if args.config.exists():
        logger.info(f"config: {args.config}")
    apply_config(args, config)

    args.urls = resolve_urls(args.urls, config)
    if not args.urls:
        logger.error(
            "Nothing to process. Pass one or more URLs (space-separated), add them to "
            'versurix_config.json ("urls"), or set VERSURIX_URL / VERSURIX_URLS. '
            "Use --local for audio file paths."
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp dir for audio unless --keep-audio
    if args.keep_audio:
        audio_dir = args.output_dir
    else:
        _tmpdir = tempfile.mkdtemp()
        audio_dir = Path(_tmpdir)

    audio_path: Optional[Path] = None
    t_start = time.monotonic()

    try:
        # Process each URL or local path
        for url in args.urls:
            log_section("Track")
            if args.local:
                log_detail(url)
                t0 = time.monotonic()
                audio_path, info = load_local_audio(url)
                source_ref = str(audio_path)
                title = str(info.get("title", audio_path.stem))
                uploader = str(info.get("uploader", "local"))
                duration = int(info.get("duration", 0))
                video_id = str(info.get("id", audio_path.stem))
                log_detail(f"{audio_path.name}  ({time.monotonic() - t0:.1f}s)")
            else:
                log_detail(url)
                log_section("Download")
                t_dl = time.monotonic()
                audio_path, info = download_audio(url, audio_dir)
                source_ref = url
                title = str(info.get("title", "Unknown"))
                uploader = str(info.get("uploader") or info.get("channel", "Unknown"))
                duration = int(info.get("duration", 0))
                video_id = str(info["id"])
                log_detail(
                    f"{audio_path.name}  ·  {uploader}  ·  {duration}s  ·  {time.monotonic() - t_dl:.1f}s"
                )

            transcribe_path = audio_path
            if args.isolate_vocals:
                log_section("Separate vocals")
                transcribe_path = isolate_vocals_demucs(audio_path, audio_dir)
            elif args.enhance_vocals:
                log_section("Enhance  (FFmpeg EQ)")
                transcribe_path = enhance_vocals_ffmpeg(audio_path, audio_dir)

            log_section("Transcribe")
            t0 = time.monotonic()
            result = transcribe_audio(
                transcribe_path, args.language, args.model, args.verbose, args.whisper
            )
            raw_text = result["text"].strip()
            segments = result.get("segments", [])
            detected_lang = str(result.get("language", args.language))
            log_detail(
                f"{args.model.split('/')[-1]}  ·  {detected_lang}  ·  {len(segments)} segments  ·  {time.monotonic() - t0:.1f}s"
            )

            # Join segments with newlines so each phrase is on its own line
            if segments:
                final_text: str = "\n".join(seg["text"].strip() for seg in segments)
            else:
                final_text = raw_text

            log_section("Wrote")
            stem: str = output_stem(title, video_id, audio_path)
            written: List[Path] = []

            if args.output_format in ("txt", "both"):
                txt_name: str = f"{stem}.txt"
                txt_path: Path = args.output_dir / txt_name

                metadata_header: str = f"// {title}\n// {source_ref}\n\n"
                full_content: str = metadata_header + final_text
                
                txt_path.write_text(full_content, encoding="utf-8")
                written.append(txt_path)

            if args.output_format in ("srt", "both") and segments:
                srt_path: Path = args.output_dir / f"{stem}.srt"
                srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
                written.append(srt_path)
            elif args.output_format in ("srt", "both") and not segments:
                logger.warning("no segment timestamps available, skipping SRT output")

            for p in written:
                log_detail(str(p.name))

            # Clean up temp audio after each URL if not keeping
            if not args.keep_audio and audio_path and audio_path.exists():
                audio_path.unlink()
                logger.debug(f"Cleaned up temporary audio file: {audio_path}")

        log_section("Done")
        _elapsed = time.monotonic() - t_start
        if len(args.urls) == 1:
            log_detail(f"{_elapsed:.1f}s")
        else:
            log_detail(f"{len(args.urls)} tracks  ·  {_elapsed:.1f}s")

    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error("Please report this issue on GitHub with the full error message")
        sys.exit(1)


if __name__ == "__main__":
    main()
