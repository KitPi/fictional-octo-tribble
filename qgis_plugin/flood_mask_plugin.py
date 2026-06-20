# flood_mask_plugin.py
import os
import tempfile

import numpy as np

# import rasterio
import requests
from PIL import Image
from qgis.core import (
    Qgis,
    QgsProject,
    QgsRasterBlock,
    QgsRasterDataProvider,
    QgsRasterFileWriter,
    QgsRasterLayer,
)
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

# from rasterio.transform import from_origin
# from sentinel1_extractor import FloodMaskModel
# from TypedQueue import TypedQueue
# from ..utils import ImageRequest


def send_single_request(url, data):
    response = requests.post(url, json=data, timeout=60)
    response.raise_for_status()
    return response.json()


class FloodMaskPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.backend_api = None
        # self.FloodModel = FloodMaskModel

    def initBackendAPI(self, api_addr):
        try:
            response = requests.get(f"http://{api_addr}/health")
            # Returns a URL with the processing ID if the API is available, otherwise returns None
            if response.status_code == 200:
                return api_addr
                # proc_id = response.json().get("processing_id", [])
                # if proc_id:
                #    return f"{api_addr}/{proc_id[0]}"
            return None

        except Exception as e:
            # self.iface.messageBar().pushMessage(e)
            return None

    def initGui(self):
        self.action = QAction("Generate Flood Mask", self.iface.mainWindow())
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Flood Analysis", self.action)
        self.action.triggered.connect(self.run)

    def unload(self):
        # self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Flood Analysis", self.action)
        del self.action

    # def save_raster(self, data, output_path, transform=None, crs=None):
    # Create a temporary file to store the output
    # with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
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

    def run(self):
        self.iface.messageBar().pushMessage("Hello from Plugin")

        # app = QApplication.instance()
        # app.setStyleSheet(".QWidget {color: blue; background-color: yellow;}")
        # Connect to the Backend API

        api_addr, ok = QInputDialog.getText(
            None, "Backend API", "Enter Backend API Address:port"
        )

        # api_addr = "127.0.0.1:8000"
        # ok = True

        self.iface.messageBar().pushMessage(f"ip-address: {api_addr}")

        if ok:
            self.backend_api = self.initBackendAPI(api_addr)
        else:
            QMessageBox.critical(
                None, "Error", "Failed to connect obtain Backend IP Address"
            )
            return

        if not self.backend_api:
            QMessageBox.critical(
                None, "Error", f"Failed to connect to Backend API at {api_addr}"
            )
            return

        self.iface.messageBar().pushMessage("Selecting Images ...")

        # Get input Sentinel-1 Raster
        input_path, _ = QFileDialog.getOpenFileName(
            None, "Select Sentinel-1 Image", "", "GeoTIFF Files (*.tif)"
        )
        if not input_path:
            return

        # self.iface.messageBar().pushMessage("Selecting Output Directory ...")

        # Get output directory
        # output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")

        # if not output_dir:
        #     QMessageBox.critical(None, "Error", "Output Directory not loaded ...")
        #     return

        # self.iface.messageBar().pushMessage("Selecting Model ...")
        # model, _ = QComboBox().addItems(["FloodModel"])
        # if not model:
        #    QMessageBox.critical(None, "Error", "Model not selected.")
        #    return

        models = ["FloodMask", "Mining", "Option 3"]
        model, ok = QInputDialog.getItem(
            None, "Title", "Select an option:", models, editable=False
        )
        if ok:
            self.iface.messageBar().pushMessage(f"Model Selected: {model}")

        self.iface.messageBar().pushMessage("Loading Raster")

        rlayer = QgsRasterLayer(input_path, "Sentinel1 Layer")
        if not rlayer.isValid():
            QMessageBox.critical(None, "Error", "Invalid raster layer.")
            return

        dp = rlayer.dataProvider()

        w, h, xt = dp.xSize(), dp.ySize(), dp.extent()

        block1 = dp.block(1, xt, w, h)
        block2 = dp.block(2, xt, w, h)

        # QgsProject.instance().addMapLayer(rlayer)

        try:
            # with rasterio.open(input_path) as img:
            #
            # vv = np.nan_to_num(rlayer.read(1),  )

            # vv = block1.as_numpy()
            # dtype = dtype_map.get(gdal_dtype, np.float32)
            vv = np.nan_to_num(
                np.frombuffer(bytes(block1.data()), dtype=np.float32).reshape((h, w))
            )

            # vh = block2.as_numpy()
            # dtype = dtype_map.get(gdal_dtype, np.float32)
            vh = np.nan_to_num(
                np.frombuffer(bytes(block2.data()), dtype=np.float32).reshape((h, w))
            )

            self.iface.messageBar().pushMessage("Sending raster to backend ...")

            response_data = send_single_request(
                url=f"http://{self.backend_api}/{model}",
                data={"vv": vv.tolist(), "vh": vh.tolist()},
            )

            self.iface.messageBar().pushMessage("Processed Raster received ...")

            # convert raster from list[[]] to QgsRasterBlock
            arr = np.nan_to_num(np.array(response_data, dtype=np.float32))
            rasterBlock = QgsRasterBlock(
                Qgis.DataType.Float32, rlayer.width(), rlayer.height()
            )
            rasterBlock.setData(arr.tobytes())

            # save raster to disk
            output_path = os.path.join(tempfile.gettempdir(), "flood_mask_output.tif")
            writer = QgsRasterFileWriter(output_path)
            provider = writer.createOneBandRaster(
                Qgis.DataType.Float32,
                rlayer.width(),
                rlayer.height(),
                rlayer.extent(),
                rlayer.crs(),
            )
            provider.writeBlock(rasterBlock, 1)
            del provider

            # load saved raster into QGIS
            output_layer = QgsRasterLayer(output_path, "Flood Mask")
            QgsProject.instance().addMapLayer(output_layer)
            # save_raster(raster, output_dir)
            return

        except Exception as e:
            raise e

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
