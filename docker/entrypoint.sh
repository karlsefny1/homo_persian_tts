#!/bin/sh
set -eu

APP_DIR=/app
BERT_DIR=/opt/bert

case "${1:-}" in
    test)
        shift
        exec python -m unittest discover -s "$APP_DIR/tests" -v "$@"
        ;;
    shell)
        exec /bin/sh
        ;;
    python)
        shift
        exec python "$@"
        ;;
    "")
        set -- --help
        ;;
esac

exec python "$APP_DIR/persian_chatterbox_tts.py" \
    --clf "$BERT_DIR/homograph_clf" \
    --lexicon "$APP_DIR/homograph_lexicon.json" \
    "$@"
