# qgis-venv-setup.sh
# Creates a Python 3.13 virtualenv and adds system dist-packages + QGIS python dir to it.
# Usage: ./qgis-venv-setup.sh [venv-path]
# Default venv-path: ./qgis-venv

VENV_PATH="${1:-./qgis-venv}"
PYTHON_BIN="${2:-python3.13}"
PYVER="$($PYTHON_BIN -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.13 or pass a different interpreter as second arg." >&2
  exit 2
fi

# create venv
"$PYTHON_BIN" -m venv "$VENV_PATH"

# upgrade packaging tools inside venv
# Use a subshell to run pip from the created venv
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

# ensure site-packages path exists
SITEPACK_DIR="$VENV_PATH/lib/python$PYVER/site-packages"
mkdir -p "$SITEPACK_DIR"

# write .pth file to expose system dist-packages and QGIS python dir
PTH_FILE="$SITEPACK_DIR/system-dist-packages.pth"
cat > "$PTH_FILE" <<EOF
/usr/lib/python3/dist-packages
/usr/share/qgis/python
EOF

echo "Virtualenv created at: $VENV_PATH"
echo "To start using it, run:"
echo "  source \"$VENV_PATH/bin/activate\""
echo
echo "Verify with:"
echo "  python -V"
echo "  python -c \"import sip, PyQt6.sip; import qgis.core; print(qgis.core.Qgis.QGIS_VERSION)\""

exit 0