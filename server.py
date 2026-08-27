"""
Persian TTS UI — FastAPI wrapper around the fine-tuned Chatterbox checkpoint.

Runs INSIDE the chatterbox-finetuning environment: activate that repo's .venv, then
    CHATTERBOX_DIR=/mnt/data/monib/chatterbox-finetuning python server.py
It chdirs into the toolkit repo (its code uses ./relative paths), loads inference.py
once (model into VRAM at startup), then serves the UI on 0.0.0.0:8200.
"""
import os
import re
import sys
import types
import tempfile
import threading

CHATTERBOX_DIR = os.environ.get("CHATTERBOX_DIR", "/mnt/data/monib/chatterbox-finetuning")
PORT = int(os.environ.get("PORT", "8200"))
MAX_CHARS = 20000
SENT_SPLIT = re.compile(r"(?<=[.?!؟…؛])\s+")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

os.chdir(CHATTERBOX_DIR)
sys.path.insert(0, CHATTERBOX_DIR)

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Load the toolkit's inference.py as a module ───────────────────────────────
# Its model setup may live at top level or under `if __name__ == "__main__"`, so we
# exec it with __name__ set to "__main__" after shrinking TEXT_TO_SAY to one word
# (the import-time self-test costs a few seconds instead of a full paragraph).
print("[boot] loading Chatterbox engine (this takes a minute)...")
_src = open("inference.py", encoding="utf-8").read()
_src = re.sub(r'^(\s*)TEXT_TO_SAY = .*$', r'\1TEXT_TO_SAY = "سلام"', _src, count=1, flags=re.M)
_mod = types.ModuleType("cb_inference")
_mod.__dict__["__name__"] = "__main__"
_mod.__dict__["__file__"] = os.path.join(CHATTERBOX_DIR, "inference.py")
exec(compile(_src, "inference.py", "exec"), _mod.__dict__)

def _need(name):
    v = _mod.__dict__.get(name)
    if v is None:
        raise RuntimeError(
            f"inference.py did not expose '{name}' at module level. "
            "Run: grep -n '^def \\|^if __name__\\|engine =' inference.py  and adjust server.py."
        )
    return v

ENGINE = _need("engine")
GEN = _need("generate_sentence_audio")
AUDIO_PROMPT = _need("AUDIO_PROMPT")
PARAMS = _mod.__dict__.get("PARAMS") or {}
print("[boot] engine ready.")

LOCK = threading.Lock()  # one generation at a time — a single T4 serializes anyway

def _to_np(chunk):
    if hasattr(chunk, "cpu"):
        chunk = chunk.cpu().numpy()
    return np.asarray(chunk, dtype=np.float32).squeeze()

def synthesize(text: str) -> str:
    sentences = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    pieces, sr = [], 24000
    for i, sent in enumerate(sentences):
        out = GEN(ENGINE, sent, AUDIO_PROMPT, **PARAMS)
        if out is None:
            raise RuntimeError(f"generation failed on sentence {i + 1}: {sent[:60]}")
        sr, chunk = out
        pieces.append(_to_np(chunk))
        pieces.append(np.zeros(int(sr * 0.25), dtype=np.float32))  # 250ms gap
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sr)
    return path

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Persian TTS")

class TtsIn(BaseModel):
    text: str

@app.get("/api/health")
def health():
    return {"ok": True, "busy": LOCK.locked()}

@app.post("/api/tts")
def tts(body: TtsIn):
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > MAX_CHARS:
        return JSONResponse({"error": f"text too long (max {MAX_CHARS} chars)"}, status_code=400)
    if not LOCK.acquire(blocking=False):
        return JSONResponse({"error": "busy — another generation is running, try again shortly"}, status_code=429)
    try:
        path = synthesize(text)
        return FileResponse(path, media_type="audio/wav", filename="tts.wav",
                            background=None, headers={"Cache-Control": "no-store"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e) or "generation failed"}, status_code=500)
    finally:
        LOCK.release()

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
