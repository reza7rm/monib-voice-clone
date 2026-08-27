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

## Auth & history

- Login: set `TTS_USER` (default `admin`) and `TTS_PASS` (**required**) in the environment.
  Sessions are cookie-based and live in server memory — a restart signs everyone out.
- Every generation is saved under `history/` (wav + json) and listed in the UI with
  play / download / delete. Set `HISTORY_DIR` to move storage (put it on the big disk).

## Run as a service

`/etc/systemd/system/persian-tts.service`:

```ini
[Unit]
Description=Persian TTS UI
After=network.target

[Service]
User=aiclient
WorkingDirectory=/mnt/data/monib/persian-tts-ui
Environment=CHATTERBOX_DIR=/mnt/data/monib/chatterbox-finetuning
Environment=TTS_USER=admin
Environment=TTS_PASS=change-me
ExecStart=/mnt/data/monib/chatterbox-finetuning/.venv/bin/python server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now persian-tts`

Notes: one generation at a time (HTTP 429 when busy); a failed sentence is retried
once then skipped (flagged in history); temp-free — audio goes straight to history.
