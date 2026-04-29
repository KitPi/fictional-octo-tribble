#!/usr/bin/env bash
set -euo pipefail

# First deactivate conda if active
# if command -v conda &> /dev/null; then
#     echo "Deactivating conda..."
#     conda deactivate || true
# fi

# Completely clean environment - remove all Anaconda references
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu" #:/usr/lib/qgis"

PYTHON=$(which python3)
#VENV= ".venv2" #"${VENV:-$HOME/venvs/qgis-venv}"
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create virtual environment if needed
#if [ ! -d "$VENV" ]; then
#    echo "Creating venv at $VENV"
#    $PYTHON -m venv "$VENV"
#    "$VENV/bin/pip" install --upgrade pip pytest
#    "$VENV/bin/pip" install -r "$PLUGIN_DIR/requirements.txt"
#fi

export PYTHONPATH="/usr/share/qgis/python/qgis:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Activate virtual environment
source "$PLUGIN_DIR/.venv2/bin/activate"

echo "Final PATH: $PATH"
echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Using Python: $(which python3)"

# Run tests
cd "$PLUGIN_DIR"
pytest -q test_runner.py