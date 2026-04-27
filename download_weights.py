"""Download model weights at build time so they're baked into the Docker image."""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_DIR = Path.home() / ".opf" / "privacy_filter"


def main():
    print(f"Downloading openai/privacy-filter weights to {MODEL_DIR}")
    snapshot_download(
        repo_id="openai/privacy-filter",
        local_dir=str(MODEL_DIR),
        allow_patterns=["original/*"],
    )

    # The opf package expects files at the top level, not in original/
    original_dir = MODEL_DIR / "original"
    if original_dir.is_dir():
        for path in original_dir.iterdir():
            dest = MODEL_DIR / path.name
            if not dest.exists():
                shutil.move(str(path), str(dest))
        original_dir.rmdir()

    # Verify
    assert (MODEL_DIR / "config.json").is_file(), "Missing config.json"
    assert any(MODEL_DIR.glob("*.safetensors")), "Missing safetensors weights"
    print(f"Weights downloaded and verified at {MODEL_DIR}")


if __name__ == "__main__":
    main()
