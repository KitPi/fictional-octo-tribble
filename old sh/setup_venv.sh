# setup_venv.sh
#!/bin/bash
set -euo pipefail
1. Create virtual environment with system site packages

VENV_DIR=".venv"
QGIS_PYTHON=(whichpython3)QGISP​YTHONP​ATH=(python3 -c "import qgis; print(qgis.file)" | sed 's|/qgis/init.py||')
Create venv

$QGIS_PYTHON -m venv --system-site-packages $VENV_DIR
2. Create qgis.pth file

echo "$QGIS_PYTHON_PATH" > $VENV_DIR/qgis.pth
echo "$QGIS_PYTHON_PATH/plugins" >> $VENV_DIR/qgis.pth
3. Create sitecustomize.py

mkdir -p $VENV_DIR/lib/python*/site-packages
cat > $VENV_DIR/lib/python*/site-packages/sitecustomize.py << 'EOF'
import os
os.add_dll_directory("/usr/lib/x86_64-linux-gnu")
os.add_dll_directory("/usr/lib/qgis")
os.add_dll_directory("/usr/lib/x86_64-linux-gnu/qt6")
EOF
4. Update pyvenv.cfg

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
sed -i "s|home = .|home = /usr|" $VENV_DIR/pyvenv.cfg
sed -i "s|executable = .|executable = /usr/bin/python3|" $VENV_DIR/pyvenv.cfg
sed -i "s|version = .*|version = $PYTHON_VERSION|" $VENV_DIR/pyvenv.cfg
5. Activate and install dependencies

source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # if you have one