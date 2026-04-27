"""Cog predictor for OpenAI Privacy Filter (PII detection and redaction)."""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# Set tiktoken cache dir BEFORE importing tiktoken anywhere.
# This must match the path used at build time so the cached BPE file is reused.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/root/.tiktoken_cache")

from cog import BasePredictor, Input


def log(msg):
    """Print to both stdout and stderr with flush, to maximize chance of capture."""
    line = f"[setup] {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model onto GPU. Wraps in try/except to log full traceback."""
        try:
            self._setup()
        except BaseException as e:
            log(f"SETUP FAILED: {type(e).__name__}: {e}")
            tb = traceback.format_exc()
            for line in tb.splitlines():
                log(line)
            raise

    def _setup(self):
        log(f"Python: {sys.version}")
        log(f"TIKTOKEN_CACHE_DIR={os.environ.get('TIKTOKEN_CACHE_DIR')}")
        log(f"cwd={os.getcwd()}")

        log("Importing torch...")
        import torch

        log(f"torch={torch.__version__}, CUDA available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"CUDA device: {torch.cuda.get_device_name(0)}")

        log("Importing tiktoken...")
        import tiktoken

        log(f"tiktoken={tiktoken.__version__}")

        log("Importing safetensors...")
        import safetensors

        log(f"safetensors={safetensors.__version__}")

        # Verify tiktoken cache directory contents
        cache_dir = Path(os.environ.get("TIKTOKEN_CACHE_DIR", ""))
        if cache_dir.is_dir():
            log(f"tiktoken cache contents at {cache_dir}:")
            for f in sorted(cache_dir.iterdir()):
                log(f"  {f.name}: {f.stat().st_size} bytes")
        else:
            log(f"tiktoken cache dir does not exist: {cache_dir}")

        # Test tiktoken encoding before doing anything else
        log("Testing tiktoken.get_encoding('o200k_base')...")
        try:
            enc = tiktoken.get_encoding("o200k_base")
            log(f"tiktoken encoding loaded OK, n_vocab={enc.n_vocab}")
            log(f"  test encode: {enc.encode('hello world')}")
        except BaseException as e:
            log(f"tiktoken FAILED: {type(e).__name__}: {e}")
            raise

        # Download model weights
        checkpoint_dir = Path.home() / ".opf" / "privacy_filter"
        config_path = checkpoint_dir / "config.json"

        if not config_path.is_file():
            log(f"Downloading model weights to {checkpoint_dir}...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                "openai/privacy-filter",
                local_dir=str(checkpoint_dir),
                allow_patterns=["original/*"],
            )
            original = checkpoint_dir / "original"
            if original.is_dir():
                log("Promoting files from original/ subdir")
                for p in original.iterdir():
                    dest = checkpoint_dir / p.name
                    if not dest.exists():
                        shutil.move(str(p), str(dest))
                original.rmdir()
            log("Download complete")
        else:
            log(f"Using existing checkpoint at {checkpoint_dir}")

        # Inspect checkpoint
        with open(config_path) as f:
            config = json.load(f)
        log(f"Config: encoding={config.get('encoding')}, num_hidden_layers={config.get('num_hidden_layers')}, vocab_size={config.get('vocab_size')}")
        log("Checkpoint files:")
        for f in sorted(checkpoint_dir.iterdir()):
            if f.is_file():
                log(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

        # Load OPF model
        log("Importing opf._api.OPF...")
        from opf._api import OPF

        log("Constructing OPF instance...")
        self.model = OPF(device="cuda", output_mode="typed")

        log("Warming up model with redact()...")
        self.model.redact("warmup")

        log("Setup complete!")

    def predict(
        self,
        text: str = Input(description="Text to scan for PII"),
        output_mode: str = Input(
            description="'typed' preserves category labels like PRIVATE_PERSON, 'redacted' collapses all to a single REDACTED label",
            choices=["typed", "redacted"],
            default="typed",
        ),
        decode_mode: str = Input(
            description="'viterbi' for coherent span boundaries (recommended), 'argmax' for raw per-token prediction",
            choices=["viterbi", "argmax"],
            default="viterbi",
        ),
    ) -> dict:
        """Detect and redact PII in text.

        Returns a structured result with the original text, detected spans
        (with labels, positions, and text), and the redacted text with
        placeholders like [PRIVATE_PERSON], [PRIVATE_EMAIL], etc.
        """
        from opf._api import RedactionResult

        self.model.set_output_mode(output_mode)
        self.model.set_decode_mode(decode_mode)
        result = self.model.redact(text)
        if isinstance(result, RedactionResult):
            return result.to_dict()
        return {"redacted_text": result}
