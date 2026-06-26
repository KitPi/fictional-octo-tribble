# GIS Remote Sensing Toolkit

----

## 1. Sentinel 1 Floodwater Detection

This tool uses the Sentinel 1 satellite imagery to detect floodwater. It is based on the work of [Sentinel 1 Floodwater Detection](https://github.com/geosolutions-it/sentinel1-flood-detection).
The tool has 3 components: a Frontend which integrates with QGIS installations as a plugin, a Backend which uses Ray serve and PyTorch to deploy a Modified ResNet50 classifier for processing InSAR Rasters, and a Training script that can be used train models on new and updated datasets. The tool is designed to be used in a cloud environment, but can also be used locally.

### 1.1. Frontend
The Frontend is a QGIS plugin that allows users to select a Sentinel 1 image and send it to the Backend for processing. The plugin is written in Python and uses the QGIS API to interact with the user interface. The plugin is designed to be used in a cloud environment, but can also be used locally.

### 1.2. Backend
The Backend is a Ray serve application that uses PyTorch to deploy a Modified ResNet50 classifier for processing InSAR Rasters. The Backend is designed to be used in a cloud environment, but can also be deployed locally. The Backend is designed to be extensible so that new models can be added for future tasks (e.g. Agricultural, Mining, Drough detection, etc.) and run simultaneously on the same backend hardware.

### 1.3. Training
The Training script PyTorch to train models on new and updated datasets. Using the Sentinel1Floods11 dataset, Modified ResNet50 models can be trained on consumer grade GPUs to detect floodwater intrusion in diverse environments. The Sentinel1Floods11 dataset has 446 instances of Flood Water rasters with corresponding flood labels. The Modified ResNet50 models achieve 61.75% IoU (Intersection over Union) accuracy which is good enough for a demo, newer model architectures and datasets containing more spectral data can achieve 88.0%+ accuracy.

## Installation

1. Clone this repository `git clone https://github.com/KitPi/fictional-octo-tribble.git`
2. Navigate to training `cd training`
3. Create a venv `python -m venv .train`
4. Activate venv `source .train/bin/activate` 
5. Install requirements `pip install -r requirements_train.txt`
6. Train model `python Train.py`
7. Deactivate venv `deactivate`

8. Navigate to backend `cd ../app`
9. Create a venv `python -m venv .app`
10. Activate venv `source .app/bin/activate`
11. Install requirements `pip install -r requirements.txt`

13. Install the plugin to QGIS *QGIS* -> *Plugins* -> *Manage and Install Plugins* -> *Install from ZIP* -> *qgis_plugin.zip*

## Running the app

### Start the backend
1. Navigate to app directory `cd app`
2. Create a venv `python -m venv .app`
3. Activate venv `source .app/bin/activate`
4. Navigate to source directory `cd ..`
5. Serve the backend `serve run app.app:FloodModelApp`

### Inferencing from the frontend
6. Open QGIS 
7. Open Sentinel1 Rasters
8. Run ToA Corrections and adjustments such as outlined in [Sentinel 1 Floodwater Detection](https://github.com/geosolutions-it/sentinel1-flood-detection)
7. Start the plugin:  *Plugins* -> *Flood Analysis* -> *Generate Flood Mask*
8. Enter back end API Address and Port: (typically `127.0.0.1:8000`)
9. Select Raster Layers
10. Select Model
11. Run Inferencing to Generate Flood Mask

## Todo:
 - [ ] Clip input rasters into *512 x 512* chunks to optimize network thoughput. Send whole rasters over the network is not feasible.
