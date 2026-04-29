#!/bin/bash
set -euo pipefail

# 1. Create virtual environment
VENV_DIR=".venv2"
python3 -m venv --system-site-packages $VENV_DIR

# 2. Activate and install PyQt6
source $VENV_DIR/bin/activate
pip install --upgrade pip
#pip install PyQt6

# 3. Set up QGIS paths
QGIS_PYTHON_PATH=$(python3 -c "import qgis; print(qgis.__file__)" | sed 's|/qgis/__init__.py||')
echo "$QGIS_PYTHON_PATH" > $VENV_DIR/qgis.pth
echo "$QGIS_PYTHON_PATH/plugins" >> $VENV_DIR/qgis.pth

# 4. Create sitecustomize.py (fixed version)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
SITE_PACKAGES_DIR="$VENV_DIR/lib/python$PYTHON_VERSION/site-packages"
mkdir -p "$SITE_PACKAGES_DIR"
cat > "$SITE_PACKAGES_DIR/sitecustomize.py" << 'EOF'
import sys
sys.path.append('/usr/share/qgis/python/qgis')
sys.path.append('/usr/share/qgis/python')
sys.path.append('/usr/share/qgis/python/plugins')
EOF

#pip install -r requirements.txt

echo "Virtual environment setup complete. Activate with: source $VENV_DIR/bin/activate"
