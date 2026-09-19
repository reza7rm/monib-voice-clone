# Pocket-TTS Farsi v2 — zero-shot Monib voice test

Tests whether `mehdi-hf/pocket-tts-farsi-v2` (109M, CPU-only, CC-BY-NC) can clone
the Monib voice from a 5-second clip with **no training**. It reads Persian far
better than raw-text models because input goes through the Homo-GE2PE
phonemizer first.

## Setup on the T4 (one time, separate venv — does not touch the training venvs)

```bash
cd /mnt/data/monib/persian-tts-ui   # wherever this repo is checked out
git pull

python3 -m venv /mnt/data/monib/pocket-venv
source /mnt/data/monib/pocket-venv/bin/activate
pip install --upgrade pip
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" \
    transformers scipy soundfile sentencepiece huggingface_hub
```

(CPU torch is fine — everything here runs without the GPU.)

## Run

```bash
cd pocket_test
python zero_shot_test.py --ref /mnt/data/monib/chatterbox-finetuning/speaker_reference/2.wav
```

Better: pick the cleanest ~5s Monib clip you can find (no music/background,
one continuous sentence) — output mimics the prompt's acoustic quality:

```bash
python zero_shot_test.py --ref /mnt/data/monib/MyTTSDataset/wavs/XXXXXX.wav
```

Custom text:

```bash
python zero_shot_test.py --ref clip.wav --text "جمله‌ی دلخواه"
python zero_shot_test.py --ref clip.wav --file my_sentences.txt   # one per line
```

Outputs land in `pocket_out/` (one wav per sentence, 24kHz) plus
`phonemes.txt` showing what the G2P produced for each line.

## What to judge

1. **Voice similarity** — does it sound like Monib? (This is the unknown; the
   Persian reading quality is already known to be good.)
2. Try the words/sentences that the Chatterbox fine-tune mispronounced.
3. If a word sounds wrong, check its line in `phonemes.txt` — a G2P mistake
   looks different from a TTS mistake.

## Verdict paths

- **Similarity good** → we can wire this into the studio UI as an engine
  option (CPU, so it can even run alongside training).
- **Voice close but not tight** → fine-tune path exists
  (`training/configs/finetune.yaml` in the pocket-tts fork) using the existing
  12,797 Monib clips; needs forced alignment + phonemization prep.
- **Voice wrong** → stay with the Chatterbox v2 adapter.

⚠️ License: CC-BY-NC-4.0 — **non-commercial**, including any fine-tune of it.
