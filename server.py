"""
Persian TTS UI — FastAPI wrapper around the fine-tuned Chatterbox checkpoint.

Run inside the chatterbox-finetuning venv:
    TTS_USER=admin TTS_PASS=secret CHATTERBOX_DIR=/mnt/data/monib/chatterbox-finetuning python server.py

Auth: username/password (env TTS_USER / TTS_PASS) → HttpOnly session cookie.
Sessions live in memory — a server restart signs everyone out, which is fine here.
Every generation is saved to HISTORY_DIR (wav + json sidecar) and browsable in the UI.
"""
import json
import os
import re
import secrets
import sys
import time
import types
import threading
import uuid

CHATTERBOX_DIR = os.environ.get("CHATTERBOX_DIR", "/mnt/data/monib/chatterbox-finetuning")
PORT = int(os.environ.get("PORT", "8200"))
AUTH_USER = os.environ.get("TTS_USER", "admin")
AUTH_PASS = os.environ.get("TTS_PASS", "")
MAX_CHARS = 20000
SENT_SPLIT = re.compile(r"(?<=[.?!؟…؛])\s+")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
HISTORY_DIR = os.environ.get("HISTORY_DIR", os.path.join(BASE_DIR, "history"))
os.makedirs(HISTORY_DIR, exist_ok=True)

if not AUTH_PASS:
    print("FATAL: set TTS_PASS (and optionally TTS_USER) in the environment.")
    sys.exit(1)

os.chdir(CHATTERBOX_DIR)
sys.path.insert(0, CHATTERBOX_DIR)

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Engine ────────────────────────────────────────────────────────────────────
print("[boot] loading Chatterbox engine (this takes a minute)...")
_src = open("inference.py", encoding="utf-8").read()
_mod = types.ModuleType("cb_inference")
_mod.__dict__["__file__"] = os.path.join(CHATTERBOX_DIR, "inference.py")
exec(compile(_src, "inference.py", "exec"), _mod.__dict__)

def _need(name):
    v = _mod.__dict__.get(name)
    if v is None:
        raise RuntimeError(f"inference.py did not expose '{name}'")
    return v

import torch
_device = "cuda" if torch.cuda.is_available() else "cpu"
_loader = _need("load_finetuned_engine_lora") if _mod.__dict__.get("IS_LORA", True) \
    else _need("load_finetuned_engine_full")
ENGINE = _loader(_device)
GEN = _need("generate_sentence_audio")
AUDIO_PROMPT = _need("AUDIO_PROMPT")
PARAMS = _mod.__dict__.get("PARAMS") or {}
print(f"[boot] engine ready on {_device}.")

LOCK = threading.Lock()
SESSIONS = set()

def _to_np(chunk):
    if hasattr(chunk, "cpu"):
        chunk = chunk.cpu().numpy()
    return np.asarray(chunk, dtype=np.float32).squeeze()

def synthesize_to_history(text: str) -> dict:
    sentences = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    pieces, sr, skipped = [], 24000, 0
    for i, sent in enumerate(sentences):
        out = GEN(ENGINE, sent, AUDIO_PROMPT, **PARAMS)
        if out is None:  # one retry — sampling occasionally misfires
            out = GEN(ENGINE, sent, AUDIO_PROMPT, **PARAMS)
        if out is None:
            skipped += 1
            print(f"[tts] SKIPPED sentence {i + 1} after retry: {sent[:80]}")
            continue
        sr, chunk = out
        pieces.append(_to_np(chunk))
        pieces.append(np.zeros(int(sr * 0.25), dtype=np.float32))
    if not pieces:
        raise RuntimeError("no audio generated")
    audio = np.concatenate(pieces)
    item_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    sf.write(os.path.join(HISTORY_DIR, item_id + ".wav"), audio, sr)
    meta = {
        "id": item_id,
        "text": text,
        "created": int(time.time()),
        "duration": round(len(audio) / sr, 1),
        "sentences": len(sentences),
        "skipped": skipped,
    }
    with open(os.path.join(HISTORY_DIR, item_id + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta

# ── Auth helpers ──────────────────────────────────────────────────────────────
def authed(request: Request) -> bool:
    return request.cookies.get("tts_session") in SESSIONS

def deny():
    return JSONResponse({"error": "unauthorized"}, status_code=401)

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Persian TTS")

class LoginIn(BaseModel):
    username: str
    password: str

class TtsIn(BaseModel):
    text: str

@app.post("/api/login")
def login(body: LoginIn):
    ok = secrets.compare_digest(body.username, AUTH_USER) and secrets.compare_digest(body.password, AUTH_PASS)
    if not ok:
        time.sleep(1)  # blunt brute-force damper
        return JSONResponse({"error": "نام کاربری یا رمز اشتباه است"}, status_code=401)
    token = secrets.token_urlsafe(32)
    SESSIONS.add(token)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("tts_session", token, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp

@app.post("/api/logout")
def logout(request: Request):
    SESSIONS.discard(request.cookies.get("tts_session"))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("tts_session")
    return resp

@app.get("/api/me")
def me(request: Request):
    return {"authed": authed(request)}

@app.get("/api/health")
def health():
    return {"ok": True, "busy": LOCK.locked()}

@app.post("/api/tts")
def tts(body: TtsIn, request: Request):
    if not authed(request):
        return deny()
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > MAX_CHARS:
        return JSONResponse({"error": f"text too long (max {MAX_CHARS} chars)"}, status_code=400)
    if not LOCK.acquire(blocking=False):
        return JSONResponse({"error": "یک تولید دیگر در حال اجراست — کمی بعد دوباره امتحان کنید"}, status_code=429)
    try:
        return {"item": synthesize_to_history(text)}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e) or "generation failed"}, status_code=500)
    finally:
        LOCK.release()

@app.get("/api/history")
def history(request: Request):
    if not authed(request):
        return deny()
    items = []
    for fn in os.listdir(HISTORY_DIR):
        if fn.endswith(".json"):
            try:
                items.append(json.load(open(os.path.join(HISTORY_DIR, fn), encoding="utf-8")))
            except Exception:
                pass
    items.sort(key=lambda x: x.get("created", 0), reverse=True)
    return {"items": items[:200]}

@app.get("/api/history/{item_id}/audio")
def history_audio(item_id: str, request: Request):
    if not authed(request):
        return deny()
    safe = re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", item_id)
    path = os.path.join(HISTORY_DIR, item_id + ".wav")
    if not safe or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav", filename=f"tts-{item_id}.wav")

@app.delete("/api/history/{item_id}")
def history_delete(item_id: str, request: Request):
    if not authed(request):
        return deny()
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", item_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    for ext in (".wav", ".json"):
        try:
            os.remove(os.path.join(HISTORY_DIR, item_id + ext))
        except OSError:
            pass
    return {"ok": True}

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
