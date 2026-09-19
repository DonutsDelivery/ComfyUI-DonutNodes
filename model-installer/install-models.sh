#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for candidate in "$SCRIPT_DIR/../venv/bin/python" "$SCRIPT_DIR/../.venv/bin/python"; do
    if [ -x "$candidate" ]; then
        exec "$candidate" "$SCRIPT_DIR/install_models.py" "$@"
    fi
done
exec python3 "$SCRIPT_DIR/install_models.py" "$@"
