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
            return None

        except Exception as e:
            raise e

    def initGui(self):
        self.action = QAction("Generate Flood Mask", self.iface.mainWindow())
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Flood Analysis", self.action)
        self.action.triggered.connect(self.run)

    def unload(self):
        # self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Flood Analysis", self.action)
        del self.action

    def run(self):
        self.iface.messageBar().pushMessage("Hello from Plugin")

        api_addr, ok = QInputDialog.getText(
            None, "Backend API", "Enter Backend API Address:port"
        )

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

        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [l for l in layers if isinstance(l, QgsRasterLayer)]

        if not raster_layers:
            QMessageBox.information(
                None, "No Rasters", "No raster layers loaded in the project."
            )
            return

        items = [l.name() for l in raster_layers]
        name, ok = QInputDialog.getItem(
            None, "Select Raster", "Choose a raster layer:", items, editable=False
        )
        if not ok:
            return

        rlayer = next(l for l in raster_layers if l.name() == name)

        models = ["FloodMask", "Mining", "Option 3"]
        model, ok = QInputDialog.getItem(
            None, "Title", "Select an option:", models, editable=False
        )
        if ok:
            self.iface.messageBar().pushMessage(f"Model Selected: {model}")

        self.iface.messageBar().pushMessage("Loading Raster")

        # rlayer = QgsRasterLayer(input_path, "Sentinel1 Layer")
        if not rlayer.isValid():
            QMessageBox.critical(None, "Error", "Invalid raster layer.")
            return

        dp = rlayer.dataProvider()

        w, h, xt = dp.xSize(), dp.ySize(), dp.extent()

        block1 = dp.block(1, xt, w, h)
        block2 = dp.block(2, xt, w, h)

        # QgsProject.instance().addMapLayer(rlayer)

        try:
            vv = np.nan_to_num(
                np.frombuffer(bytes(block1.data()), dtype=np.float32).reshape((h, w))
            )

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


if __name__ == "__console__":
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    plugin = FloodMaskPlugin(None)
    app.exitQgis()
