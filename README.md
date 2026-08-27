# Persian TTS UI

Minimal web UI + API for the fine-tuned Chatterbox Persian voice.
Paste text → get a WAV in the cloned voice. Sentence-by-sentence generation
with 250ms gaps, served by the same machine that holds the model.

## Run (on the GPU server)

```bash
# into the SAME venv the fine-tuning repo uses (it has torch etc.)
source /mnt/data/monib/chatterbox-finetuning/.venv/bin/activate
pip install -r requirements.txt

CHATTERBOX_DIR=/mnt/data/monib/chatterbox-finetuning python server.py
```

Open http://SERVER-IP:8200 — the model loads once at startup (about a minute),
then each request generates sentence by sentence (a few seconds per sentence on a T4).

- `POST /api/tts` `{"text": "..."}` → `audio/wav`
- `GET /api/health` → `{ok, busy}`
- One generation at a time (HTTP 429 when busy).
- The voice reference is whatever `speaker_reference/2.wav` is in the toolkit repo.

## Keep it running

```bash
screen -S ttsui
source /mnt/data/monib/chatterbox-finetuning/.venv/bin/activate
CHATTERBOX_DIR=/mnt/data/monib/chatterbox-finetuning python server.py
# detach: Ctrl-A then D
```
