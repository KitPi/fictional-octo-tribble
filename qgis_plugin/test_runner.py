# test_runner.py
import os
import pytest
import tempfile
import numpy as np
from qgis.core import QgsApplication, QgsRasterLayer, QgsProject
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session", autouse=True)
def qgis_app():
    # set QGIS prefix path to /usr (adjust if your QGIS is elsewhere)
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()
    yield app
    app.exitQgis()

def test_import_plugin():    
    """Test that the plugin can be imported without errors"""
    try:
        from flood_mask_plugin import FloodMaskPlugin  
    except Exception as e:
        pytest.fail(f"Failed to import plugin: {e}")

def test_plugin_basic_functionality():
    # Example: call a function in your plugin that doesn't require a running QGIS GUI
    from flood_mask_plugin import FloodMaskPlugin
    result = FloodMaskPlugin.example_function(2, 3)
    assert result == 5
