import tempfile

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

net = models.segmentation.fcn_resnet50(
    pretrained=False, num_classes=2, pretrained_backbone=False
)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

FloodModelPath = "checkpoints/Sen1Floods11_663_0.5874795913696289.cp"


class FloodMaskModel(nn.Module):
    def __init__(self):
        super(FloodMaskModel, self).__init__()
        self.model = net
        self.model.load_state_dict(torch.load(FloodModelPath, map_location=self.device))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loadModel(self):
        self.model.to(self.device)

    def forward(self, x):
        im = self.preprocess_image(x)
        return self.postprocess_output(self.model(im)["out"])

    def forward_batch(self, batch):
        return [self.forward(x) for x in batch]

    def preprocess_image(self, image_path):
        # Read the input image
        with rasterio.open(image_path) as src:
            im1 = Image.from_array(src.read(1))  # Read the first band
            im2 = Image.from_array(src.read(2))  # Read the second band

            norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
            im = torch.stack(
                [
                    transforms.ToTensor()(im1).squeeze(),
                    transforms.ToTensor()(im2).squeeze(),
                ]
            )
            im = norm(im)
            # Normalize to [0, 1]
            # image = image.astype(np.float32) / 65535.0  # Assuming 16-bit image
            # image is already floating32
            return im

        return None

    def postprocess_output(self, output):
        # Get predictions
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.squeeze().numpy()

        # Convert to binary numpy mask (1 for flood, 0 for no flood)
        flood_mask = (predicted == 1).astype(np.uint8) * 255

        return flood_mask
