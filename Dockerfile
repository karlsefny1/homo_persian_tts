# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS bert

ARG BERT_REPO=https://github.com/karlsefny1/bert_persian_homograph_classifier.git
ARG BERT_REF=665291323c3b127ad806d82b94a1a556bca65840

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system \
    && GIT_LFS_SKIP_SMUDGE=1 git clone "${BERT_REPO}" /opt/bert \
    && cd /opt/bert \
    && git checkout "${BERT_REF}" \
    && git lfs pull --include="homograph_clf/model.pt" \
    && rm -rf /opt/bert/.git

FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/opt/bert \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-chatterbox.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.6.0 torchaudio==2.6.0 \
    && python -m pip install -r requirements-chatterbox.txt

COPY --from=bert /opt/bert /opt/bert
COPY persian_chatterbox_tts.py homograph_text.py homograph_preprocessor.py homograph_lexicon.json ./
COPY tests ./tests
COPY samples ./samples
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && python -m unittest discover -s tests -v \
    && python persian_chatterbox_tts.py --dry-run \
        --text "او اتاق را ترک کرد." \
        --clf /opt/bert/homograph_clf \
        --lexicon ./homograph_lexicon.json

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
