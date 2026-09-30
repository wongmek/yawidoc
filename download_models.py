#!/usr/bin/env python
"""
download_models.py — one-time download of the models used by the
"pattani_thai_translator" character.

  1. Systran/faster-whisper-small (CTranslate2 Whisper that recognizes Malay
     out of the box) -> ./whisper-thai-pattani-finetuned
     (this is the folder the character yaml's faster_whisper.model_path points to;
      replace its contents with your fine-tuned CT2 export later if you train one)
  2. facebook/nllb-200-distilled-600M -> Hugging Face cache under ./models
     (run_server.py sets HF_HOME=./models, so the server finds the cached copy)

Run:  uv run python download_models.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
# Match run_server.py: models and tokenizer caches live under ./models
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    ct2_dir = ROOT / "whisper-thai-pattani-finetuned"

    print("== 1/2: Systran/faster-whisper-small (CT2) ->", ct2_dir)
    snapshot_download(
        repo_id="Systran/faster-whisper-small",
        local_dir=str(ct2_dir),
    )
    files = sorted(p.name for p in ct2_dir.glob("*") if p.is_file())
    print("   files:", ", ".join(files))
    if "model.bin" not in files or "config.json" not in files:
        print("❌ CT2 model folder is missing model.bin / config.json", file=sys.stderr)
        sys.exit(1)

    print("== 2/2: facebook/nllb-200-distilled-600M -> ./models/nllb-200-distilled-600M")
    nllb_dir = ROOT / "models" / "nllb-200-distilled-600M"
    # NOTE: downloaded as local_dir (plain files) because the default HF cache
    # layout needs OS symlinks, which are unavailable on Windows without
    # Developer Mode / admin privileges.
    snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        local_dir=str(nllb_dir),
        allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"],
    )
    if not (nllb_dir / "model.safetensors").exists():
        # repo ships only .bin weights -> fetch those instead
        snapshot_download(
            repo_id="facebook/nllb-200-distilled-600M",
            local_dir=str(nllb_dir),
            allow_patterns=["pytorch_model.bin"],
        )
    if not any(
        (nllb_dir / w).exists()
        for w in ("model.safetensors", "pytorch_model.bin")
    ):
        print(f"❌ No model weights found in {nllb_dir}", file=sys.stderr)
        sys.exit(1)

    print("✅ Model download complete.")


if __name__ == "__main__":
    main()
