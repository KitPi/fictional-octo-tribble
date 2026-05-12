import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI
from ray import serve

from utils import *

app = FastAPI()

FloodModelPath = "checkpoints/Sen1Floods11_663_0.5874795913696289.cp"


@serve.deployment
@serve.ingress(app)
class FloodModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(FloodModelPath, map_location=self.device)
        self.model.eval()
        self.norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.5)
    async def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        batch_tensor = (
            torch.stack([torch.from_numpy(img) for img in images])
            .to(self.device)
            .float()
        )

        batch_tensor = self.norm(batch_tensor)

        with torch.no_grad():
            outputs = self.model(batch_tensor)

        return outputs.cpu().numpy().tolist()

    @app.post("/")
    async def handle_request(self, request: list[ImageRequest]):
        images_array = [
            torch.stack([np.array(req.vv), np.array(req.vh)]) for req in request
        ]

        predictions = await self.predict(images_array)

        return [{"prediction": pred} for pred in predictions]
