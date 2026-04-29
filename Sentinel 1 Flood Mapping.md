
Sentinel 1 Flood Mapping

[Sen1Floods](https://github.com/cloudtostreet/Sen1Floods11/blob/master/Train.ipynb)

/home/kit/Documents/QGIS/FloodMapping


## Setup

```bash
python3.13 -m venv qgis-venv
source qgis-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

```bash
PYVER=3.13
echo "/usr/lib/python3/dist-packages" > "$VIRTUAL_ENV/lib/python$PYVER/site-packages/system-dist-packages.pth"
echo "/usr/share/qgis/python" >> "$VIRTUAL_ENV/lib/python$PYVER/site-packages/system-dist-packages.pth"
```

```bash
python -m pytest
```