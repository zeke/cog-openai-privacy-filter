"""Cog predictor for OpenAI Privacy Filter (PII detection and redaction)."""

import json
import logging
import shutil
import sys
import traceback
from pathlib import Path

from cog import BasePredictor, Input

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("privacy-filter")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model onto GPU."""
        try:
            self._setup()
        except Exception as e:
            log.error(f"Setup failed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            raise

    def _setup(self):
        import torch
        import tiktoken
        import safetensors

        log.info(f"Python: {sys.version}")
        log.info(f"torch: {torch.__version__}")
        log.info(f"tiktoken: {tiktoken.__version__}")
        log.info(f"safetensors: {safetensors.__version__}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            log.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

        # Download weights explicitly with logging
        checkpoint_dir = Path.home() / ".opf" / "privacy_filter"
        config_path = checkpoint_dir / "config.json"

        if not config_path.is_file():
            log.info(f"Downloading model weights to {checkpoint_dir}...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                "openai/privacy-filter",
                local_dir=str(checkpoint_dir),
                allow_patterns=["original/*"],
            )
            original = checkpoint_dir / "original"
            if original.is_dir():
                log.info("Promoting files from original/ subdir")
                for p in original.iterdir():
                    dest = checkpoint_dir / p.name
                    if not dest.exists():
                        shutil.move(str(p), str(dest))
                original.rmdir()
            log.info("Download complete")
        else:
            log.info(f"Using existing checkpoint at {checkpoint_dir}")

        # Inspect the checkpoint
        assert config_path.is_file(), f"Missing config.json at {config_path}"
        with open(config_path) as f:
            config = json.load(f)
        log.info(f"Checkpoint config keys: {sorted(config.keys())}")
        log.info(f"  encoding: {config.get('encoding')}")
        log.info(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
        log.info(f"  d_model: {config.get('d_model')}")

        # List checkpoint files
        log.info("Checkpoint files:")
        for f in sorted(checkpoint_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / 1e6
                log.info(f"  {f.name}: {size_mb:.1f} MB")

        # Verify tiktoken encoding works before loading model
        encoding_name = config.get("encoding")
        if encoding_name:
            log.info(f"Testing tiktoken encoding: {encoding_name}")
            try:
                enc = tiktoken.get_encoding(encoding_name)
                log.info(f"tiktoken encoding OK (n_vocab={enc.n_vocab})")
            except Exception as e:
                log.error(f"tiktoken.get_encoding({encoding_name!r}) failed: {e}")
                # Don't raise here, let opf try its own loading path

        # Load the OPF model
        log.info("Loading OPF model...")
        from opf._api import OPF

        self.model = OPF(device="cuda", output_mode="typed")
        log.info("Warming up with sample input...")
        self.model.redact("warmup")
        log.info("Setup complete")

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
