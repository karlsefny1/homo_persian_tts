#!/usr/bin/env python3
"""Persian text -> local BERT homograph diacritizer -> Persian Chatterbox -> WAV."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import unicodedata

from homograph_text import strip_diacritics

HERE = Path(__file__).resolve().parent
PERSIAN_MODEL = "Thomcles/Chatterbox-TTS-Persian-Farsi"
BASE_MODEL = "ResembleAI/chatterbox"
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
HARAKAT = set("ًٌٍَُِّْ")


def split_text(text, max_chars=250):
    """Split after BERT inference, preserving every character and whole words.

    Prefer sentence boundaries. Oversized individual words fail explicitly;
    truncating text or splitting a letter from its combining marks is unsafe.
    """
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    pieces = re.findall(r"\S+\s*|\s+", text)
    chunks, current = [], ""
    for piece in pieces:
        if len(piece.rstrip()) > max_chars:
            raise ValueError("A word exceeds --max-chars; shorten it before synthesis")
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = ""
        current += piece
        if re.search(r"[.!?؟؛…]\s*$", current):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def check_diacritic_tokens(tokenizer, text):
    """Refuse silent loss of BERT's vowels at the actual TTS tokenizer.

    Token IDs changing is necessary, but does not prove the acoustic model
    learned those marks. Listening comparisons are still required.
    """
    marks = Counter(c for c in unicodedata.normalize("NFKD", text) if c in HARAKAT)
    if not marks:
        return
    ids = tokenizer.encode(text, language_id=None)
    plain_ids = tokenizer.encode(strip_diacritics(text), language_id=None)
    if ids == plain_ids:
        raise RuntimeError("TTS tokenizer discarded the homograph diacritics")
    decoded = tokenizer.tokenizer.decode(ids, skip_special_tokens=False)
    recovered = Counter(c for c in decoded if c in HARAKAT)
    if any(recovered[c] < count for c, count in marks.items()):
        raise RuntimeError("TTS tokenizer cannot preserve all homograph diacritics")


def load_chatterbox(device="auto", weights=None, model_id=PERSIAN_MODEL,
                    revision="main", cache_dir=None):
    import torch
    import perth
    if perth.PerthImplicitWatermarker is None:
        try:
            from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Perth watermarking could not be imported. Run "
                "`bash sandbox_chatterbox.sh setup` to repair dependencies. "
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError("Perth watermarking is unavailable; restart Python after repairing dependencies")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Do not download the unused 2 GB multilingual T3: initialize its structure
    # and load the Persian checkpoint directly, with strict shape/key checking.
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.tokenizers import MTLTokenizer

    def base_file(name):
        return hf_hub_download(BASE_MODEL, name, revision=BASE_REVISION,
                               cache_dir=cache_dir)

    if weights:
        checkpoint = Path(weights).expanduser().resolve(strict=True)
    else:
        checkpoint = hf_hub_download(model_id, "t3_fa.safetensors",
                                     revision=revision, cache_dir=cache_dir)
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    t3.to(device).eval()
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(base_file("ve.pt"), map_location="cpu", weights_only=True))
    ve.to(device).eval()
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(base_file("s3gen.pt"), map_location="cpu", weights_only=True))
    s3gen.to(device).eval()
    # Prevent an unnecessary lazy download by the multilingual tokenizer.
    base_file("Cangjie5_TC.json")
    tokenizer = MTLTokenizer(base_file("grapheme_mtl_merged_expanded_v1.json"))
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device)


class PersianChatterboxPipeline:
    """Keep BERT and TTS resident for repeated requests. Supply both for testing."""

    def __init__(self, diacritizer=None, tts=None, *, clf=HERE / "homograph_clf",
                 lexicon=HERE / "homograph_lexicon.json", bert_device="cpu",
                 min_conf=0.8, max_chars=250, **tts_options):
        self.diacritizer = diacritizer
        self.tts = tts
        self.clf, self.lexicon = clf, lexicon
        self.bert_device, self.min_conf = bert_device, min_conf
        self.max_chars, self.tts_options = max_chars, tts_options

    def prepare(self, text):
        if not text.strip():
            raise ValueError("Text must not be empty")
        if self.diacritizer is None:
            from homograph_preprocessor import HomographDiacritizer
            self.diacritizer = HomographDiacritizer(
                str(self.clf), str(self.lexicon), device=self.bert_device,
                min_conf=self.min_conf)
        prepared, decisions = self.diacritizer.analyze(text)
        if strip_diacritics(prepared) != strip_diacritics(text):
            raise RuntimeError("Classifier changed characters other than diacritics")
        return prepared, decisions

    def render(self, prepared, reference, *, seed=42, temperature=0.8,
               cfg_weight=0.5, exaggeration=0.5):
        """Render already prepared text; normally call synthesize() instead."""
        import numpy as np
        import torch

        if not prepared.strip():
            raise ValueError("Text must not be empty")
        reference = Path(reference).expanduser().resolve(strict=True)
        chunks = split_text(prepared, self.max_chars)
        if self.tts is None:
            self.tts = load_chatterbox(**self.tts_options)
        for chunk in chunks:
            check_diacritic_tokens(self.tts.tokenizer, chunk)
        self.tts.prepare_conditionals(str(reference), exaggeration=exaggeration)
        audio = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            random.seed(seed + i)
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + i)
            with torch.inference_mode():
                # This fine-tune uses no language prefix. Upstream has no [fa]
                # support; using [ar] changes the intended conditioning.
                wav = self.tts.generate(
                    chunk, language_id=None, temperature=temperature,
                    cfg_weight=cfg_weight, exaggeration=exaggeration)
            wav = wav.detach().cpu()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if wav.ndim != 2 or wav.shape[0] != 1 or wav.shape[-1] == 0 or not torch.isfinite(wav).all():
                raise RuntimeError("TTS returned invalid or empty mono audio")
            if audio:
                audio.append(torch.zeros((1, int(0.18 * self.tts.sr)), dtype=wav.dtype))
            audio.append(wav)
        return torch.cat(audio, dim=-1), self.tts.sr

    def synthesize(self, text, reference, **options):
        prepared, decisions = self.prepare(text)
        wav, sr = self.render(prepared, reference, **options)
        return wav, sr, prepared, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path, help="UTF-8 Persian text")
    parser.add_argument("--ref", type=Path, help="clean native Persian reference WAV")
    parser.add_argument("--out", type=Path, default=Path("chatterbox.wav"))
    parser.add_argument("--report", type=Path, help="save text, decisions and settings as JSON")
    parser.add_argument("--clf", type=Path, default=HERE / "homograph_clf")
    parser.add_argument("--lexicon", type=Path, default=HERE / "homograph_lexicon.json")
    parser.add_argument("--min-conf", type=float, default=0.8)
    parser.add_argument("--bert-device", default="cpu")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--weights", type=Path, help="existing t3_fa.safetensors")
    parser.add_argument("--model-id", default=PERSIAN_MODEL)
    parser.add_argument("--revision", default="main", help="Persian checkpoint revision; pin for reproducibility")
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--max-chars", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="BERT only; no TTS imports or downloads")
    parser.add_argument("--ab", action="store_true", help="also save original text as *_plain.wav")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_conf <= 1:
        parser.error("--min-conf must be between 0 and 1")
    if args.max_chars < 20:
        parser.error("--max-chars must be at least 20")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not args.dry_run and (args.ref is None or not args.ref.is_file()):
        parser.error("--ref must name an existing reference audio file")
    if args.out.suffix.lower() != ".wav":
        parser.error("--out must have a .wav extension")
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    pipeline = PersianChatterboxPipeline(
        clf=args.clf, lexicon=args.lexicon, bert_device=args.bert_device,
        min_conf=args.min_conf, max_chars=args.max_chars, device=args.device,
        weights=args.weights, model_id=args.model_id, revision=args.revision,
        cache_dir=args.cache_dir)
    prepared, decisions = pipeline.prepare(text)
    print("original:", text)
    print("to TTS  :", prepared)
    if args.verbose:
        print(json.dumps(decisions, ensure_ascii=False, indent=2))
    options = dict(seed=args.seed, temperature=args.temperature,
                   cfg_weight=args.cfg_weight, exaggeration=args.exaggeration)
    if not args.dry_run:
        import soundfile as sf
        jobs = [(args.out, prepared)]
        if args.ab:
            jobs.insert(0, (args.out.with_name(args.out.stem + "_plain.wav"), text))
        for path, content in jobs:
            wav, sr = pipeline.render(content, args.ref, **options)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), wav.squeeze(0).numpy(), sr, subtype="PCM_16")
            print("wrote", path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(
            original=text, prepared=prepared, decisions=decisions,
            chunks=split_text(prepared, args.max_chars), model=args.model_id,
            revision=args.revision, weights=str(args.weights) if args.weights else None,
            base_revision=BASE_REVISION, settings=options, dry_run=args.dry_run),
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
