# __init__.py
def classFactory(iface):
    from .flood_mask_plugin import FloodMaskPlugin
    return FloodMaskPlugin(iface)