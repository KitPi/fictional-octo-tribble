import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ray import serve

# from .utils import *


class ImageRequest(BaseModel):
    vv: list[list[float]]
    vh: list[list[float]]


def convertBNtoGN(module, num_groups=16):
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        mod = nn.GroupNorm(
            num_groups, module.num_features, eps=module.eps, affine=module.affine
        )
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        return mod

    for name, child in module.named_children():
        module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

    return module


app = FastAPI()

net = models.segmentation.fcn_resnet50(
    weights=None, num_classes=2, weights_backbone=None
)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
net = convertBNtoGN(net)

FloodModelPath = "checkpoints/Sen1Floods11_663_0.5874795913696289.cp"


@serve.deployment
@serve.ingress(app)
class FloodModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = net
        self.model.load_state_dict(torch.load(FloodModelPath, map_location=self.device))
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

    @app.post("/")  # /floodmask/process
    async def handle_request(self, request: ImageRequest) -> list[list[float]]:
        images_array = [torch.stack([np.array(request.vv), np.array(request.vh)])]

        predictions = await self.predict(images_array)

        return predictions

    @app.get("/health")
    async def health_check(self) -> str:
        return "OK"

    @app.get("/status", response_class=HTMLResponse)
    async def index(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FloodMask Service</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 80px auto; text-align: center; }
    .status { color: green; font-size: 1.2em; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>FloodMask Service</h1>
  <p class="status">&#9989; Service is running</p>
  <p>POST <code>/</code> with <code>{"vv": [...], "vh": [...]}</code> to process imagery.</p>
  <p><a href="/health">/health</a></p>
</body>
</html>"""


FloodModelApp = FloodModel.options(
    autoscaling_config={
        "target_ongoing_requests": 50,
        "min_replicas": 1,
        "max_replicas": 10,
        "upscale_delay_s": 5,
        "downscale_delay_s": 30,
    },
    max_ongoing_requests=200,
    max_queued_requests=-1,
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,
    },
).bind()
