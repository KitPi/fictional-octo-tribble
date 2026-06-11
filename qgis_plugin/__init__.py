# __init__.py
from .flood_mask_plugin import FloodMaskPlugin


def classFactory(iface):
    return FloodMaskPlugin(iface)
