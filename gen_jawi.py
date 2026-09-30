#!/usr/bin/env python
"""One-off: generate the Jawi (Malay, Arabic script) version of the
translator greeting using the local NLLB model (tha_Thai -> zsm_Arab)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

from src.open_llm_vtuber.translate.nllb_translator import NLLBTranslator  # noqa: E402

THAI = "ต้องการให้ฉันช่วยแปลภาษาคำไหนแจ้งได้เลย"

print("Trying tha_Thai -> zsm_Arab ...")
try:
    t = NLLBTranslator(
        model_name="./models/nllb-200-distilled-600M",
        src_lang="tha_Thai",
        tgt_lang="zsm_Arab",
        device="cpu",
    )
    jawi = t.translate(THAI)
    print("Jawi (zsm_Arab):")
    print(jawi)
    print("Thai source:")
    print(THAI)
except Exception as e:
    print(f"zsm_Arab failed: {e}")
    print("Falling back to tha_Thai -> zsm_Latn (Malay, Latin script) ...")
    t = NLLBTranslator(
        model_name="./models/nllb-200-distilled-600M",
        src_lang="tha_Thai",
        tgt_lang="zsm_Latn",
        device="cpu",
    )
    print("Malay (zsm_Latn):")
    print(t.translate(THAI))
