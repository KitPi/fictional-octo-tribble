# flood_mask_plugin.py
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.PyQt.QtGui import *
from qgis.core import QgsRasterLayer, QgsProject
import rasterio
from rasterio.transform import from_origin
import tempfile
from PIL import Image



import requests

class FloodMaskPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.backend_api = None

    def initBackendAPI(self, api_addr):
        try:
            response = requests.get(f"{api_addr}/status")
            # Returns a URL with the processing ID if the API is available, otherwise returns None
            if response.status_code == 200:
                proc_id = response.json().get("processing_id", [])
                if proc_id: return f"{api_addr}/{proc_id[0]}" 
            return None
        
        except Exception as e:
            return None

    def initGui(self):
        self.action = QAction("Generate Flood Mask", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Flood Analysis", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginToMenu("&Flood Analysis", self.action)

    def run(self):
        # Connect to the Backend API
        api_addr = QInputEvent.getText(None, "Backend API", "Enter Backend API Address:port")
        self.backend_api = self.initBackendAPI(api_addr)
        if not self.backend_api:
            QMessageBox.critical(
                None, "Error", f"Failed to connect to Backend API at {api_addr}"
            )
            return


        # Get input Sentinel-1 Raster
        input_path, _ = QFileDialog.getOpenFileName(
            None, "Select Sentinel-1 Image", "", "GeoTIFF Files (*.tif)"
        )
        if not input_path:
            return


        job_ids = []

        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            None, "Select Output Directory"
        )
        if not output_dir:
            return

        # Transfer Raster to Backend API
        try:
            with open(input_path, 'rb') as f: # Path To .tif files
                files = {'file': f}
                response = requests.post(f"{self.backend_api}/process-raster", files=files)
                if response.status_code != 200:
                    raise Exception(f"API Error: {response.text}")
                output_data = response.content
                job_ids.append(response.json().get("job_id", "unknown"))
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to process image: {str(e)}")
            return
    

if __name__ == "__console__":
    from qgis.core import QgsApplication
    app = QgsApplication([], False)
    app.initQgis()
    plugin = FloodMaskPlugin(None)
    app.exitQgis()