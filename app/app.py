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

    @app.post("/floodmask/process")
    async def handle_request(self, request: ImageRequest) -> list[np.ndarray]:
        images_array = [torch.stack([np.array(request.vv), np.array(request.vh)])]

        predictions = await self.predict(images_array)

        return predictions

    @app.get("/health")
    async def health_check(self) -> str:
        return "OK"


num_cpus_per_replica = 1
num_gpus_per_replica = 1
FloodModelApp = FloodModel.options(
    autoscaling_config={
        "target_ongoing_requests": 50,
        "min_replicas": 1,
        "max_replicas": 10,
        "upscale_delay_s": 5,
        "downscale_delay_s": 30,
    },
    max_ongoing_requests=200,
    max_queued_request=-1,
    ray_actor_options={
        "num_cpus": num_cpus_per_replica,
        "num_gpus": num_gpus_per_replica,
    },
).bind()

handle = serve.run(FloodModelApp, name="FloodModelApp")
