#!/usr/bin/env python
"""
test_ws_translation.py — headless browser-equivalent test over the real
WebSocket server:

  connect -> switch to the "pattani_thai_translator" character
  (this initializes faster-whisper + NLLB through ServiceContext.load_from_config)
  -> send Malay text -> receive the translated Thai text + synthesized audio.

This exercises the exact path the browser frontend uses.

Prerequisite: the server must be running (uv run run_server.py).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

# Windows consoles default to a legacy codepage (cp874) that can't encode
# Thai text or emoji — force UTF-8 so results print cleanly.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import websocket  # websocket-client (sync)

URL = "ws://localhost:12393/client-ws"
CHARACTER_FILE = "translator_pattani_thai.yaml"
MALAY_TEXT = "Selamat pagi, saya nak pergi ke pasar untuk membeli ikan segar."
THAI = re.compile(r"[฀-๿]")


def recv_msg(ws, timeout):
    ws.settimeout(timeout)
    try:
        return json.loads(ws.recv())
    except websocket.WebSocketTimeoutException:
        return None


def preview(msg, limit=200):
    text = json.dumps(msg, ensure_ascii=False)
    if msg.get("type") == "audio" and isinstance(msg.get("audio"), str):
        text = text.replace(msg["audio"], f"<{len(msg['audio'])} chars of base64 wav>")
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> int:
    print(f"Connecting to {URL} ...")
    ws = websocket.create_connection(URL, timeout=15)
    print("Connected. Draining initial messages ...")
    t0 = time.time()
    while time.time() - t0 < 15:
        msg = recv_msg(ws, 3)
        if msg is None:
            break
        print(f"  <- {preview(msg)}")

    # ---- switch to the translator character (loads ASR + NLLB engines) ----
    print(f"\n-> switch-config: {CHARACTER_FILE}")
    ws.send(json.dumps({"type": "switch-config", "file": CHARACTER_FILE}))

    switched = False
    t0 = time.time()
    while time.time() - t0 < 180:
        msg = recv_msg(ws, 5)
        if msg is None:
            continue
        if msg.get("type") == "config-switched":
            print(f"  <- {preview(msg)}  [engines loaded]")
            switched = True
            break
        if msg.get("type") == "error":
            print(f"  <- SERVER ERROR: {preview(msg)}")
            return 2
        print(f"  <- {preview(msg)}")
    if not switched:
        print("TIMEOUT waiting for config-switched", file=sys.stderr)
        return 2

    # ---- trigger "ai-speak-signal" (what the frontend sends at
    # conversation start) and expect the bilingual greeting ----
    print("\n-> ai-speak-signal (conversation start)")
    ws.send(json.dumps({"type": "ai-speak-signal"}))

    greeting_shown = None
    greeting_audio = False
    greeting_synth = False
    t0 = time.time()
    while time.time() - t0 < 120:
        msg = recv_msg(ws, 5)
        if msg is None:
            if greeting_shown and greeting_audio and greeting_synth:
                break
            continue
        mtype = msg.get("type", "")
        if mtype == "error":
            print(f"  <- SERVER ERROR: {preview(msg)}")
            return 3
        display_text = msg.get("display_text")
        if (
            greeting_shown is None
            and isinstance(display_text, dict)
            and isinstance(display_text.get("text"), str)
            and "ต้องการให้ฉันช่วยแปล" in display_text["text"]
        ):
            greeting_shown = display_text["text"]
            print(f"  <- GREETING DISPLAY: {greeting_shown}")
        if mtype == "audio" and isinstance(msg.get("audio"), str):
            greeting_audio = True
            print(f"  <- greeting audio ({len(msg['audio'])} chars of base64 wav)")
        if mtype == "backend-synth-complete":
            greeting_synth = True
            print("  <- backend-synth-complete (greeting)")

    greeting_ok = bool(greeting_shown) and "کبو ساي" in (greeting_shown or "")
    if not greeting_ok:
        print(f"FAILED: bilingual greeting not shown correctly: {greeting_shown!r}", file=sys.stderr)
        return 4
    print("  Greeting contains both Thai and Jawi: OK")

    # ---- send Malay text like the browser's text input box ----
    print(f"\n-> text-input: {MALAY_TEXT}")
    ws.send(json.dumps({"type": "text-input", "text": MALAY_TEXT}))

    thai_text = None
    got_audio = False
    synth_complete = False
    chain_ended = False
    t0 = time.time()
    while time.time() - t0 < 240:
        msg = recv_msg(ws, 5)
        if msg is None:
            if chain_ended and synth_complete:
                break
            continue
        mtype = msg.get("type", "")
        if mtype == "error":
            print(f"  <- SERVER ERROR: {preview(msg)}")
            return 3

        if not thai_text:
            display_text = msg.get("display_text")
            if isinstance(display_text, dict) and isinstance(
                display_text.get("text"), str
            ) and THAI.search(display_text["text"]):
                thai_text = display_text["text"]
                print(f"  <- TRANSLATED THAI TEXT: {thai_text}")

        if mtype == "audio" and isinstance(msg.get("audio"), str):
            got_audio = True
            print(f"  <- audio payload ({len(msg['audio'])} chars of base64 wav)")
        if mtype == "backend-synth-complete":
            synth_complete = True
            print("  <- backend-synth-complete")
        if mtype == "control" and msg.get("text") == "conversation-chain-end":
            chain_ended = True
            print("  <- conversation-chain-end")

        if thai_text and got_audio and synth_complete:
            break

    ws.close()

    print("\n==================== WEBSOCKET TEST RESULT ====================")
    print(f"Greeting (TH+Jawi) : {greeting_shown!r}")
    print(f"Thai text received : {thai_text!r}")
    print(f"Thai audio received: {got_audio}")
    print(f"Synth complete     : {synth_complete}")
    if greeting_ok and thai_text and got_audio and synth_complete:
        print("PASSED ✅ — greeting + translation + TTS all work over the server path")
        return 0
    print("FAILED ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
