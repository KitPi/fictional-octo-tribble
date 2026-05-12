import datetime
import uuid

from fastapi import FastAPI, HTTPException

from backend import SuterbrookBackendProcessor
from utils import *

app = FastAPI()

processor = None


@app.on_event("startup")
async def startup_event():
    global processor
    processor = SuterbrookBackendProcessor()


@app.get("shutdown")
async def shutdown_event():
    if processor:
        processor.shutdown()


@app.get("/FloodModel/process")
async def process(file):
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialised")
    item = Item(time=datetime.now(), job_id=uuid.uuid4(), data=file)
    processor.FloodModelQueue.push(item)
    return {
        "message": f"Added item {item.uuid} to queue",
        "queue_size": processor.FloodModelQueue.size(),
    }
