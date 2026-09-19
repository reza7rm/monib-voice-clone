"""Zero-shot Monib voice test for Pocket-TTS Farsi v2 (mehdi-hf/pocket-tts-farsi-v2).

Clones the voice from a short reference clip (<=5s) with NO training, phonemizing
Persian text through Homo-GE2PE first (the model reads phonemes; raw Persian
script produces silence).

Usage (CPU is fine, no GPU needed):
    python zero_shot_test.py --ref /path/to/monib_clip.wav
    python zero_shot_test.py --ref clip.wav --text "متن دلخواه"
    python zero_shot_test.py --ref clip.wav --file sentences.txt

Outputs one wav per sentence into ./pocket_out/ plus phonemes.txt with the
G2P output for each line (check it when something sounds wrong).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile
import soundfile as sf

MODEL_REPO = "mehdi-hf/pocket-tts-farsi-v2"
G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"
MAX_PROMPT_SEC = 5.0  # hard training cap; longer prompts derail generation

DEFAULT_SENTENCES = [
    "سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید.",
    "خداوند در قرآن کریم می‌فرماید که با یاد او دل‌ها آرام می‌گیرد.",
    "امروز می‌خواهیم درباره‌ی صبر و شکیبایی در زندگی صحبت کنیم.",
    "سال هزار و چهارصد و پنج، بیست و پنج شهریور ماه.",
]


def load_normalizer():
    """Import normalize_fa.py shipped inside the model's HF repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_REPO, "normalize_fa.py")
    spec = importlib.util.spec_from_file_location("normalize_fa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["normalize_fa"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_phonemizer():
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    norm = load_normalizer()
    tok = AutoTokenizer.from_pretrained(G2P_REPO)
    g2p = T5ForConditionalGeneration.from_pretrained(G2P_REPO).eval()
    # G2P output -> the phoneme alphabet the TTS tokenizer was trained on
    to_phonemes = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

    def phonemise(text: str) -> str:
        text = norm.normalize_for_model(text)
        text = text.replace("؟", "").replace("?", "")
        enc = tok([text], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
        raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return raw.translate(to_phonemes)

    return phonemise


def prepare_prompt(ref_path: Path, out_dir: Path) -> Path:
    """Validate the reference clip; trim to MAX_PROMPT_SEC if longer."""
    audio, sr = sf.read(str(ref_path), always_2d=True)
    dur = len(audio) / sr
    print(f"Reference: {ref_path} ({dur:.2f}s @ {sr} Hz)")
    if dur <= MAX_PROMPT_SEC:
        return ref_path
    trimmed = out_dir / "prompt_trimmed.wav"
    sf.write(str(trimmed), audio[: int(MAX_PROMPT_SEC * sr)], sr)
    print(f"  clip is over {MAX_PROMPT_SEC}s -> trimmed copy at {trimmed}")
    return trimmed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="Reference wav of the target voice (<=5s ideal)")
    ap.add_argument("--text", help="Single Persian sentence to synthesize")
    ap.add_argument("--file", help="Text file, one Persian sentence per line")
    ap.add_argument("--out", default="pocket_out", help="Output directory")
    args = ap.parse_args()

    if args.text:
        sentences = [args.text]
    elif args.file:
        sentences = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading G2P (Homo-GE2PE)...")
    phonemise = build_phonemizer()

    print("Loading Pocket-TTS Farsi v2 (CPU)...")
    from pocket_tts import TTSModel

    tts = TTSModel.load_model(config=f"hf://{MODEL_REPO}/model.yaml")

    prompt = prepare_prompt(Path(args.ref), out_dir)
    voice_state = tts.get_state_for_audio_prompt(str(prompt))

    phoneme_log = []
    for i, sent in enumerate(sentences, 1):
        ph = phonemise(sent)
        phoneme_log.append(f"{i:02d} | {sent}\n   | {ph}")
        print(f"[{i}/{len(sentences)}] {sent}")
        print(f"        phonemes: {ph}")
        # copy_state=True (default) keeps voice_state clean for the next sentence
        audio = tts.generate_audio(voice_state, ph)
        wav = out_dir / f"{i:02d}.wav"
        scipy.io.wavfile.write(str(wav), tts.sample_rate, audio.numpy())
        print(f"        -> {wav} ({audio.shape[-1] / tts.sample_rate:.1f}s)")

    (out_dir / "phonemes.txt").write_text("\n".join(phoneme_log), encoding="utf-8")
    print(f"\nDone. Listen to the wavs in {out_dir}/ — G2P output logged to {out_dir}/phonemes.txt")


if __name__ == "__main__":
    main()
