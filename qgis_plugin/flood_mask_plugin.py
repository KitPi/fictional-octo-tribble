# flood_mask_plugin.py
import asyncio
import os
import tempfile

import aiohttp
import numpy as np

# import rasterio
import requests
from PIL import Image
from qgis.core import QgsProject, QgsRasterBlock, QgsRasterBlockType, QgsRasterLayer
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QInputEvent,
    QMessageBox,
)

# from rasterio.transform import from_origin
# from sentinel1_extractor import FloodMaskModel
# from TypedQueue import TypedQueue
from utils import *


async def send_single_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()


class FloodMaskPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.backend_api = None
        # self.FloodModel = FloodMaskModel

    def initBackendAPI(self, api_addr):
        try:
            response = requests.get(f"{api_addr}/status")
            # Returns a URL with the processing ID if the API is available, otherwise returns None
            if response.status_code == 200:
                proc_id = response.json().get("processing_id", [])
                if proc_id:
                    return f"{api_addr}/{proc_id[0]}"
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

    def save_raster(self, data, output_path, transform=None, crs=None):
        # Create a temporary file to store the output
        #with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        #    # Save the data as a GeoTIFF
        #    with rasterio.open(
        #        tmp.name,
        #        "w",
        #        driver="GTiff",
        #        height=data.shape[0],
        #        width=data.shape[1],
        #        count=1,
        #        dtype=data.dtype,
        #        crs=crs,
        #        transform=transform,
        #    ) as dst:
            #        dst.write(np.array(data, dtype=np.float32), 1)
            #
            #    # Copy the temporary file to the final destination
        #    with open(tmp.name, "rb") as src, open(output_path, "wb") as dst:
            #        dst.write(src.read())

    async def run(self):
        # Connect to the Backend API
        api_addr = QInputEvent.getText(
            None, "Backend API", "Enter Backend API Address:port"
        )
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

        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")
        if not output_dir:
            return

        model = QComboBox().addItems(["FloodModel"])
        if not model:
            return

        rlayer = QgsRasterLayer(input_path, "Sentinel1 Layer")
        if not rlayer.isValid():
            QMessageBox.critical(None, "Error", "Invalid raster layer")
            return

        QgsProject.instance().addMapLayer(rlayer)

        try:
            # with rasterio.open(input_path) as img:
            #
            # vv = np.nan_to_num(rlayer.read(1), nan=0.0)
            vv = rlayer.as_numpy(bands=1)
            # vh = np.nan_to_num(img.read(2), nan=0.0)
            vh = rlayer.as_numpy(bands=2)
            with aiohttp.ClientSession() as session:
                raster = await send_single_request(
                    session,
                    url=f"{self.backend_api}/{model}",
                    data={"vv": vv.tolist(), "vh": vh.tolist()},
                )

                rasterBlock = QgsRasterBlock(
                    QgsRasterBlockType.Float32, rlayer.width(), rlayer.height()
                )
                rasterBlock.setData(raster)
                rlayer.setData(rasterBlock)
                QgsProject.instance().addMapLayer(rlayer)
                # save_raster(raster, output_dir)

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to process image: {str(e)}")
            return

        # Transfer Raster to Backend API
        # try:
        #    with open(input_path, "rb") as f:  # Path To .tif files
        #        file_content = f.read()
        #        item = Item(time=datetime.now(), data=file_content)
        #        async with requests.post(
        #            f"{self.backend_api}/{model}/process", files=files
        #        ) as response
        #            if response.status_code != 200:
        #                raise Exception(f"API Error: {response.text}")
        #            output_data = response.content
        #            job_ids.append(response.json().get("job_id", "unknown"))
        # except Exception as e:
        #    QMessageBox.critical(None, "Error", f"Failed to process image: {str(e)}")
        #    return


if __name__ == "__console__":
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    plugin = FloodMaskPlugin(None)
    app.exitQgis()
