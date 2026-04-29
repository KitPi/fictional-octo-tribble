import requests
import queue
import uuid
import rasterio
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

q = queue.Queue()
processing_list = []
processed = []
failed = []

app = FastAPI()

# Import Models
FloodMaskModel = None

@app.post("/status")
async def get_status():
    return JSONResponse({
        "status": "Sentinel 1 Floodmapping API is available."
    })

@app.post("/process-raster")
async def process_raster(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = f"/tmp/{job_id}/input.tif"

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
        job_id = str(uuid.uuid4())
        q.put(job_id)
        return JSONResponse({
            "job_id": job_id,
            "message": "Raster received and queued for processing."
        })
    return JSONResponse({
        "error": "Failed to save the uploaded raster."
    }, status_code=500)         

@app.post("/job-status/{job_id}")
async def job_status(job_id: str): 

    if job_id in list(q.queue):
        return JSONResponse({
            "job_id": job_id,
            "status": "Queued"
        })
    if job_id in processing_list:
        return JSONResponse({
            "job_id": job_id,
            "status": "Processing"
        })
    if job_id in processed:
        return JSONResponse({
            "job_id": job_id,
            "status": "Completed"
        })
    if job_id in failed:
        return JSONResponse({
            "job_id": job_id,
            "status": "Failed"
        })
    return JSONResponse({
        "error": "Job not found."
    }, status_code=404)


@app.post("/results/{job_id}")
async def results(job_id: str):
    if job_id in list(q.queue):
        return JSONResponse({
            "job_id": job_id,
            "status": "Queued"
        })
    if job_id in processing_list:
        return JSONResponse({
            "job_id": job_id,
            "status": "Processing"
        })
    if job_id in processed:
         with open(f"/tmp/{job_id}/output.tif", 'rb') as f: # Path To .tif files
            return JSONResponse({
                "job_id": job_id,
                "status": "Completed",
                'file': f
            })
    if job_id in failed:
        return JSONResponse({
            "job_id": job_id,
            "status": "Failed"
        })
    return JSONResponse({
        "error": "Job not found."
    }, status_code=404)

import time
def working_thread():
    while True:
        if not q.empty():
            job_id = q.get()
            processing_list.append(job_id)
            
            input_path = f"/tmp/{job_id}/input.tif"
            output_path = f"/tmp/{job_id}/output.tif"

            success = process_image_test(input_path, output_path)

            if success:
                processing_list.remove(job_id)
                processed.append(job_id)
            else:
                processing_list.remove(job_id)
                failed.append(job_id)
        else:
            time.sleep(0.1)


def process_image(input_path, output_path):
    try:
        with rasterio.open(input_path) as src:
            # Simulate processing time
            time.sleep(5) # 
            # For demonstration, we just copy the input to output
            with rasterio.open(output_path, 'w', **src.meta) as dst:
                dst.write(src.read())
        return True
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False
    

def process_image_test(input_path, output_path):
    try:
        with rasterio.open("/home/kit/Documents/QGIS/RemoteSensing/test_images/Bolivia_23014_LabelHand.tif") as src:
            # Simulate processing time
            time.sleep(5) # 
            # For demonstration, we just copy the input to output
            with rasterio.open(output_path, 'w', **src.meta) as dst:
                dst.write(src.read())
        return True
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False
    
