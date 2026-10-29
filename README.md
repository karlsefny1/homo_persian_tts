# Persian Homograph-Aware Chatterbox TTS

This project wraps the Persian Chatterbox TTS fine-tune with a local BERT
homograph classifier. The classifier adds Persian vowel marks (harakat) only to
ambiguous words before synthesis, so the TTS model pronounces homographs such as
`گل` ("gol", flower) and `گل` ("gel", mud) in the intended reading.

The pipeline is reproducible through Docker Compose:

```text
Persian text
    -> BERT homograph classifier
    -> validated diacritized text
    -> Persian Chatterbox TTS
    -> 24 kHz mono WAV + JSON decision report
```

## What was done

- Added a text-to-WAV command-line pipeline in `persian_chatterbox_tts.py`.
- Integrated the BERT homograph classifier from
  `karlsefny1/bert_persian_homograph_classifier` before the TTS step.
- Loaded only the Persian Chatterbox T3 checkpoint, avoiding the unused ~2 GB
  multilingual T3 weights.
- Added guards so the classifier may change diacritics but may not rewrite
  letters, words, or punctuation.
- Added a tokenizer check that fails if the TTS tokenizer silently drops the
  classifier's vowel marks.
- Added safe chunking for long input while preserving every character, word, and
  combining mark.
- Added deterministic seeds, chunk stitching, and invalid/silent-audio checks.
- Added an optional A/B mode that renders both plain and diacritized text for
  listening comparison.
- Added a Dockerfile, Docker Compose services, a GPU override, a sample
  reference WAV, and unit tests that do not download models.
- Restored the runtime BERT source files and regenerated
  `homograph_lexicon.json` from `review.csv` using `build_lexicon.py`.

## What changed compared with the original Chatterbox model

The acoustic model itself is not retrained here. The changes are in how the
model is loaded and how text reaches it:

| Area | Original Chatterbox | This project |
| --- | --- | --- |
| Persian support | Upstream Chatterbox does not provide a `fa` language ID | Uses the Persian T3 fine-tune `Thomcles/Chatterbox-TTS-Persian-Farsi` |
| Model loading | Convenience loader may download the full multilingual T3 | Initializes T3 and loads only `t3_fa.safetensors`, with strict key/shape checks |
| Language conditioning | Uses a language prefix/ID | Uses `language_id=None`, matching the Persian fine-tune |
| Text preparation | Raw text is passed to TTS | Local BERT classifier adds only homograph diacritics |
| Safety | No homograph rewrite guard | Refuses non-diacritic changes and refuses tokenizer mark loss |
| Long input | Basic or no chunking | Sentence-aware chunking with character preservation and explicit failures |
| Outputs | WAV only | WAV plus optional JSON report and optional plain-vs-diacritized A/B files |
| Runtime | Manual Python environment | Docker image and Compose services with cached Hugging Face models |

In short: the base Chatterbox architecture and vocoder are unchanged. The
Persian T3 checkpoint, the BERT preprocessing stage, the validation rules, and
the packaging are the project-specific additions.

The Dockerfile pins the BERT classifier repository to commit
`665291323c3b127ad806d82b94a1a556bca65840`. The upstream Chatterbox base assets
are pinned to `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. The Persian checkpoint
defaults to `main`; for a frozen deployment, pass `--revision <commit>` to
`persian_chatterbox_tts.py`.

## BERT model locations

The classifier weights are not stored in this repository. During the Docker
build, the BERT repository is cloned at a pinned commit and
`homograph_clf/model.pt` is fetched through Git LFS. The image contains:

- `/opt/bert/homograph_clf/`: `config.json`, `meta.json`, tokenizer files, and
  `model.pt`
- `/app/homograph_lexicon.json`: the reading lexicon copied from this repository

`docker/entrypoint.sh` passes those two paths as `--clf` and `--lexicon` to the
pipeline. For local runs, point `--clf` at your clone's `homograph_clf` directory
and `--lexicon` at this repository's `homograph_lexicon.json`.

## Requirements

- Docker with Compose v2 or newer.
- About 10 GB of free disk space for the image, BERT weights, and Hugging Face
  model cache.
- A Hugging Face account with access to the gated Persian checkpoint:
  <https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi>
- Optional: an NVIDIA GPU and a recent driver for faster inference.

The first synthesis downloads roughly 3.2 GB of Hugging Face model files. The
BERT weights (about 456 MB) are downloaded into the image at build time.

## Docker quick start

### 1. Accept the model terms and create a token

Open the model page above, accept its terms, then create a read token at
<https://huggingface.co/settings/tokens>.

```bash
cp .env.example .env
# Put the token in .env:
# HF_TOKEN=hf_...
```

Do not commit `.env`.

### 2. Build the image

```bash
docker compose build
```

The default build uses CPU PyTorch wheels. The build clones the BERT repository
at a pinned commit, downloads `homograph_clf/model.pt` through Git LFS, runs the
unit tests, and performs a BERT-only Persian smoke test.

### 3. Run the tests without downloading the TTS model

```bash
docker compose --profile tools run --rm test
```

The unit tests use mocked BERT/TTS components. They do not need Hugging Face
access, a GPU, or model downloads.

### 4. Run a BERT-only check

```bash
docker compose --profile tools run --rm bert
```

This reads `samples/example.txt`, applies the homograph classifier, and writes
`outputs/bert-check.json`. It does not load or download Chatterbox.

### 5. Synthesize the sample

```bash
docker compose run --rm tts
```

This reads `samples/example.txt`, uses `samples/reference.wav`, and writes:

- `outputs/demo.wav`
- `outputs/demo.json`

The first run downloads the Persian T3 checkpoint and the upstream Chatterbox
voice encoder, S3Gen, and tokenizer files. Later runs reuse the named
`hf-cache` volume.

## Use your own text and voice

The Compose file mounts `./samples` at `/input` and `./outputs` at `/outputs`.

```bash
docker compose run --rm tts \
  --text "او برای رسیدن به مقصدش از مسیر کوتاه‌تری دور زد." \
  --ref /input/reference.wav \
  --out /outputs/custom.wav \
  --report /outputs/custom.json \
  --verbose
```

To synthesize a file:

```bash
docker compose run --rm tts \
  --file /input/example.txt \
  --ref /input/reference.wav \
  --out /outputs/from-file.wav
```

To compare plain and diacritized input:

```bash
docker compose run --rm tts \
  --file /input/example.txt \
  --ref /input/reference.wav \
  --out /outputs/compare.wav \
  --ab
```

This writes `outputs/compare_plain.wav` and `outputs/compare.wav`.

For the reference clip, use 6-15 seconds of clean, mono, native Persian speech
with little noise or music. Replace `samples/reference.wav` or pass another path
under `/input`.

## GPU

Build with CUDA PyTorch wheels and enable the GPU override:

```bash
printf '\nTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124\n' >> .env
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm tts
```

The BERT classifier runs on CPU by default. To use a GPU for BERT as well, pass
`--bert-device cuda` and add a GPU device request for the relevant service.

If a CUDA wheel does not match your driver, choose the matching index from
<https://pytorch.org/get-started/locally/> and set `TORCH_INDEX_URL` accordingly.

## Local installation without Docker

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchaudio==2.6.0
python -m pip install -r requirements-chatterbox.txt
```

Clone the BERT assets separately:

```bash
git lfs install
git clone https://github.com/karlsefny1/bert_persian_homograph_classifier.git \
  /opt/bert_persian
git -C /opt/bert_persian lfs pull --include=homograph_clf/model.pt
```

Run the tests and a BERT-only check:

```bash
python -m unittest discover -s tests -v
python persian_chatterbox_tts.py \
  --dry-run \
  --text "گل قرمز در گلدان است." \
  --clf /opt/bert_persian/homograph_clf \
  --lexicon ./homograph_lexicon.json
```

Run synthesis:

```bash
export HF_TOKEN=hf_...
python persian_chatterbox_tts.py \
  --file samples/example.txt \
  --ref samples/reference.wav \
  --out outputs/local.wav \
  --report outputs/local.json \
  --clf /opt/bert_persian/homograph_clf \
  --lexicon ./homograph_lexicon.json
```

## Tests and smoke checks

Unit tests:

```bash
docker compose --profile tools run --rm test
```

They cover:

- classifier output reaching TTS unchanged;
- rejection of word rewrites;
- blank input not loading models;
- chunking preserving text and combining marks;
- oversized words failing explicitly;
- tokenizer diacritic loss being rejected.

BERT-only check:

```bash
docker compose --profile tools run --rm bert
```

Full smoke test:

```bash
docker compose run --rm tts
ls -lh outputs/demo.wav outputs/demo.json
```

A successful run prints the original text, the text sent to TTS, the classifier
decisions in verbose mode, and the output paths. The report JSON includes the
prepared text, decisions, chunk list, model IDs, base revision, and generation
settings.

## Troubleshooting

**401 or 403 while downloading the Persian model**

Accept the terms on the model page and make sure `HF_TOKEN` in `.env` is valid.
Restart the command after updating the token.

**`model.pt` is still a small Git LFS pointer**

Run `git lfs pull --include=homograph_clf/model.pt` in the BERT clone. The
Docker build does this automatically.

**The build runs out of disk**

The final image plus Hugging Face cache is several gigabytes. Free space or
move Docker's data root to a larger volume.

**CPU synthesis is slow**

This is expected. Build the CUDA variant and enable `compose.gpu.yaml` if you
have a compatible NVIDIA GPU.

**`TTS tokenizer discarded the homograph diacritics`**

The tokenizer does not preserve the marks produced by the classifier. Do not
disable the check; use the pinned tokenizer/model revision or inspect the
tokenizer files.

**A homograph is pronounced incorrectly**

Check the decision report and the classifier confidence. Raising `--min-conf`
makes the system more conservative. Do not lower it blindly: applying a wrong
reading is worse than leaving a word undiacritized.

**Output is clipped, silent, or noisy**

Use a clean 6-15 second reference clip and verify it plays correctly. The
pipeline rejects empty or non-finite TTS output, but it cannot judge recording
quality.

## Licenses and responsible use

- The Persian Chatterbox checkpoint is published as CC-BY-NC-4.0 on Hugging
  Face, so it is not suitable for commercial use without separate permission.
  Verify the current model card before use.
- Upstream Chatterbox is MIT licensed. The BERT base model and fine-tuned
  classifier have their own terms; review those repositories before
  redistribution.
- Chatterbox output includes a Perth implicit watermark.
- Only clone voices you have permission to use, and disclose synthetic audio
  where required.

## Repository layout

```text
persian_chatterbox_tts.py      Persian text -> BERT -> Chatterbox CLI
homograph_preprocessor.py      BERT homograph classifier runtime
homograph_text.py              Shared Persian text normalization
homograph_lexicon.json         Reading lexicon for homograph candidates
build_lexicon.py               Rebuilds the lexicon from review.csv
review.csv                     Lexicon review data, not needed at runtime
requirements-chatterbox.txt    Python dependencies
Dockerfile                     CPU/GPU image with BERT LFS assets
compose.yaml                   Default TTS, BERT-only, and test services
compose.gpu.yaml               Adds the NVIDIA GPU device request
docker/entrypoint.sh           Injects BERT paths and dispatches commands
samples/example.txt            Persian sample with homographs
samples/reference.wav          Bundled smoke-test reference clip
tests/                         Mocked unit tests, no model downloads
outputs/                       Generated WAV and JSON files
```
