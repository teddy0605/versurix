"""
Tests for versurix.py

Run with: pytest
Fast — no downloads, no AI inference, no external services.
"""

import argparse
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import versurix


# ---------------------------------------------------------------------------
# _format_srt_time
# ---------------------------------------------------------------------------

class TestFormatSrtTime:
    def test_zero(self):
        assert versurix._format_srt_time(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert versurix._format_srt_time(1.0) == "00:00:01,000"

    def test_sub_second(self):
        assert versurix._format_srt_time(0.5) == "00:00:00,500"

    def test_minutes(self):
        assert versurix._format_srt_time(90.0) == "00:01:30,000"

    def test_hours(self):
        assert versurix._format_srt_time(3661.0) == "01:01:01,000"

    def test_millisecond_precision(self):
        assert versurix._format_srt_time(1.123) == "00:00:01,123"

    def test_rounding(self):
        assert versurix._format_srt_time(1.9999) == "00:00:02,000"


# ---------------------------------------------------------------------------
# segments_to_srt
# ---------------------------------------------------------------------------

class TestSegmentsToSrt:
    def _seg(self, start, end, text):
        return {"start": start, "end": end, "text": text}

    def test_empty(self):
        assert versurix.segments_to_srt([]) == ""

    def test_single_segment(self):
        result = versurix.segments_to_srt([self._seg(0.0, 2.5, " Hello world")])
        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "Hello world" in result

    def test_multiple_segments_numbered(self):
        segs = [self._seg(0.0, 1.0, " One"), self._seg(1.0, 2.0, " Two"), self._seg(2.0, 3.0, " Three")]
        result = versurix.segments_to_srt(segs)
        assert "1\n" in result
        assert "2\n" in result
        assert "3\n" in result

    def test_text_is_stripped(self):
        result = versurix.segments_to_srt([self._seg(0.0, 1.0, "   padded   ")])
        lines = result.split("\n")
        text_line = next(l for l in lines if "padded" in l)
        assert text_line == "padded"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_none_returns_empty(self):
        assert versurix.load_config(None) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert versurix.load_config(tmp_path / "nonexistent.json") == {}

    def test_valid_json(self, tmp_path):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"language": "en", "model": "some/model"}))
        result = versurix.load_config(cfg)
        assert result["language"] == "en"
        assert result["model"] == "some/model"

    def test_malformed_json_returns_empty(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("{not valid json")
        assert versurix.load_config(cfg) == {}

    def test_empty_json_object(self, tmp_path):
        cfg = tmp_path / "empty.json"
        cfg.write_text("{}")
        assert versurix.load_config(cfg) == {}

# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def _parse(self, args_list):
        old = sys.argv
        sys.argv = ["versurix"] + args_list
        try:
            return versurix.parse_args()
        finally:
            sys.argv = old

    def test_no_positional_urls_allowed(self):
        args = self._parse([])
        assert args.urls == []

    def test_config_defaults_to_project_json(self):
        args = self._parse([])
        assert args.config == versurix.DEFAULT_CONFIG_PATH

    def test_single_url(self):
        args = self._parse(["https://youtube.com/watch?v=test"])
        assert args.urls == ["https://youtube.com/watch?v=test"]

    def test_multiple_urls(self):
        args = self._parse(["https://youtube.com/watch?v=a", "https://youtube.com/watch?v=b"])
        assert len(args.urls) == 2

    def test_language_default_is_none(self):
        args = self._parse(["https://youtube.com/watch?v=test"])
        assert args.language is None

    def test_language_explicit(self):
        args = self._parse(["--language", "en", "https://youtube.com/watch?v=test"])
        assert args.language == "en"

    def test_model_default_is_none(self):
        args = self._parse(["https://youtube.com/watch?v=test"])
        assert args.model is None

    def test_output_format_default_is_none(self):
        args = self._parse(["https://youtube.com/watch?v=test"])
        assert args.output_format is None

    def test_output_format_choices(self):
        for fmt in ("txt", "srt", "both"):
            args = self._parse(["--output-format", fmt, "https://youtube.com/watch?v=test"])
            assert args.output_format == fmt

    def test_output_format_invalid(self):
        with pytest.raises(SystemExit):
            self._parse(["--output-format", "docx", "https://youtube.com/watch?v=test"])

    def test_enhance_vocals_flag(self):
        args = self._parse(["--enhance-vocals", "https://youtube.com/watch?v=test"])
        assert args.enhance_vocals is True

    def test_isolate_vocals_flag(self):
        args = self._parse(["--isolate-vocals", "https://youtube.com/watch?v=test"])
        assert args.isolate_vocals is True

    def test_keep_audio_flag(self):
        args = self._parse(["--keep-audio", "https://youtube.com/watch?v=test"])
        assert args.keep_audio is True

    def test_download_only_flag(self):
        args = self._parse(["--download-only", "https://youtube.com/watch?v=test"])
        assert args.download_only is True

    def test_local_flag(self):
        args = self._parse(["--local", "/tmp/song.mp3"])
        assert args.local is True

    def test_verbose_flag(self):
        args = self._parse(["--verbose", "https://youtube.com/watch?v=test"])
        assert args.verbose is True

    def test_output_dir_default_is_none(self):
        args = self._parse(["https://youtube.com/watch?v=test"])
        assert args.output_dir is None

    def test_config_path(self, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text("{}")
        args = self._parse(["--config", str(cfg), "https://youtube.com/watch?v=test"])
        assert args.config == cfg


# ---------------------------------------------------------------------------
# resolve_urls
# ---------------------------------------------------------------------------

class TestResolveUrls:
    @pytest.fixture(autouse=True)
    def _clear_versurix_url_env(self, monkeypatch):
        monkeypatch.delenv("VERSURIX_URL", raising=False)
        monkeypatch.delenv("VERSURIX_URLS", raising=False)

    def test_positional_wins_over_config(self):
        cfg = {"urls": ["https://json.only/wrong"]}
        got = versurix.resolve_urls(["https://cli/wins"], cfg)
        assert got == ["https://cli/wins"]

    def test_multiple_positional(self):
        got = versurix.resolve_urls(
            ["https://a.com/x", "https://b.com/y"],
            {},
        )
        assert got == ["https://a.com/x", "https://b.com/y"]

    def test_config_when_no_positional(self):
        got = versurix.resolve_urls([], {"urls": ["https://from/json"]})
        assert got == ["https://from/json"]

    def test_empty_when_nothing(self):
        assert versurix.resolve_urls([], {}) == []


class TestResolveUrlsEnv:
    def test_versurix_url_before_config(self, monkeypatch):
        monkeypatch.delenv("VERSURIX_URLS", raising=False)
        monkeypatch.setenv("VERSURIX_URL", "https://youtube.com/watch?v=env1")
        got = versurix.resolve_urls([], {"urls": ["https://from/json"]})
        assert got == ["https://youtube.com/watch?v=env1"]

    def test_versurix_urls_comma_separated(self, monkeypatch):
        monkeypatch.delenv("VERSURIX_URL", raising=False)
        monkeypatch.setenv("VERSURIX_URLS", "https://a.com/x,https://b.com/y")
        got = versurix.resolve_urls([], {})
        assert got == ["https://a.com/x", "https://b.com/y"]

    def test_positional_beats_env(self, monkeypatch):
        monkeypatch.setenv("VERSURIX_URL", "https://env.skip")
        got = versurix.resolve_urls(["https://cli/wins"], {})
        assert got == ["https://cli/wins"]


# ---------------------------------------------------------------------------
# output stem / sanitize
# ---------------------------------------------------------------------------

class TestSanitizeOutputStem:
    def test_keeps_unicode_and_punctuation(self):
        s = versurix.sanitize_output_stem("ME VOLVÍ P*TA (1963)")
        assert "*" in s or "VOLV" in s
        assert "/" not in s

    def test_strips_path_chars(self):
        assert "/" not in versurix.sanitize_output_stem("a/b/c")


class TestOutputStem:
    def test_prefers_audio_stem(self, tmp_path):
        p = tmp_path / "ME VOLVÍ P＊TA  - mix.mp3"
        p.write_text("x")
        stem = versurix.output_stem("ASCII Title", "id123", p)
        assert "VOLV" in stem or "mix" in stem

    def test_fallback_title(self):
        stem = versurix.output_stem("Hello (world)", "vid", Path("/no/such/file.mp3"))
        assert "Hello" in stem


class TestHubModelSnapshotsReady:
    def test_false_when_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert versurix.hub_model_snapshots_ready("mlx-community/whisper-large-v3-turbo") is False

    def test_true_when_snapshot_has_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        snap = (
            tmp_path
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--mlx-community--whisper-large-v3-turbo"
            / "snapshots"
            / "abc123"
        )
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_text("x")
        assert versurix.hub_model_snapshots_ready("mlx-community/whisper-large-v3-turbo") is True


# ---------------------------------------------------------------------------
# merge_whisper_decode
# ---------------------------------------------------------------------------

class TestMergeWhisperDecode:
    def test_empty_config_uses_defaults(self):
        merged = versurix.merge_whisper_decode({})
        assert merged == versurix.WHISPER_DECODE_DEFAULTS

    def test_partial_override(self):
        merged = versurix.merge_whisper_decode(
            {"whisper": {"no_speech_threshold": 0.5}}
        )
        assert merged["no_speech_threshold"] == 0.5
        assert merged["logprob_threshold"] == versurix.WHISPER_DECODE_DEFAULTS["logprob_threshold"]

    def test_non_dict_whisper_ignored(self):
        merged = versurix.merge_whisper_decode({"whisper": "bad"})
        assert merged == versurix.WHISPER_DECODE_DEFAULTS

    def test_optional_word_timestamps(self):
        merged = versurix.merge_whisper_decode({"whisper": {"word_timestamps": True}})
        assert merged["word_timestamps"] is True
        assert merged["no_speech_threshold"] == versurix.WHISPER_DECODE_DEFAULTS["no_speech_threshold"]

    def test_temperature_list_becomes_tuple(self):
        merged = versurix.merge_whisper_decode({"whisper": {"temperature": [0.0, 0.2]}})
        assert merged["temperature"] == (0.0, 0.2)


# ---------------------------------------------------------------------------
# apply_config
# ---------------------------------------------------------------------------

def _blank_args(**overrides):
    """Return a Namespace with all sentinel/default values, optionally overridden."""
    defaults = dict(
        language=None, model=None, output_format=None, output_dir=None,
        keep_audio=False, download_only=False, verbose=False,
        enhance_vocals=False, isolate_vocals=False, local=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestApplyConfig:
    def test_config_fills_unset_language(self):
        args = versurix.apply_config(_blank_args(), {"language": "fr"})
        assert args.language == "fr"

    def test_cli_overrides_config_language(self):
        args = versurix.apply_config(_blank_args(language="en"), {"language": "fr"})
        assert args.language == "en"

    def test_hardcoded_default_when_nothing_set(self):
        args = versurix.apply_config(_blank_args(), {})
        assert args.language == "es"
        assert args.output_format == "txt"
        assert args.output_dir == Path(versurix.DEFAULT_OUTPUT_DIR)
        assert args.model == versurix.WHISPER_DEFAULT_MODEL
        assert args.whisper == versurix.WHISPER_DECODE_DEFAULTS

    def test_whisper_from_config(self):
        args = versurix.apply_config(
            _blank_args(),
            {"whisper": {"no_speech_threshold": 0.55, "logprob_threshold": -1.5}},
        )
        assert args.whisper["no_speech_threshold"] == 0.55
        assert args.whisper["logprob_threshold"] == -1.5
        assert args.whisper["compression_ratio_threshold"] == versurix.WHISPER_DECODE_DEFAULTS[
            "compression_ratio_threshold"
        ]

    def test_config_sets_bool_flag(self):
        args = versurix.apply_config(_blank_args(), {"keep_audio": True})
        assert args.keep_audio is True

    def test_download_only_from_config(self):
        args = versurix.apply_config(_blank_args(), {"download_only": True})
        assert args.download_only is True

    def test_cli_bool_not_overridden_by_config(self):
        args = versurix.apply_config(_blank_args(isolate_vocals=True), {"isolate_vocals": False})
        assert args.isolate_vocals is True

    def test_local_from_config(self):
        args = versurix.apply_config(_blank_args(), {"local": True})
        assert args.local is True

    def test_output_dir_becomes_path(self):
        args = versurix.apply_config(_blank_args(), {"output_dir": "/tmp/lyrics"})
        assert isinstance(args.output_dir, Path)
        assert args.output_dir == Path("/tmp/lyrics")

