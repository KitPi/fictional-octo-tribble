import torch
import torch.nn as nn
import torchvision.models as models
import rasterio
from PIL import Image
from torchvision import transforms
import tempfile
import numpy as np

net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

class FloodMaskModel(nn.Module):
    def __init__(self):
        super(FloodMaskModel, self).__init__()
        self.model = net

    def forward(self, x):
        im = self.preprocess_image(x)
        return self.model(im)['out']

    def forward_batch(self, batch):
        return [self.model(x)['out'] for x in batch]
    


    def preprocess_image(self, image_path):
        # Read the input image
        with rasterio.open(image_path) as src:
            im1 = Image.from_array(src.read(1))  # Read the first band
            im2 = Image.from_array(src.read(2))  # Read the second band

            norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
            im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
            im = norm(im)
            # Normalize to [0, 1]
            #image = image.astype(np.float32) / 65535.0  # Assuming 16-bit image
            # image is already floating32 
            return im


        return None

    def postprocess_output(self, output):
        # Get predictions
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.squeeze().numpy()

        # Convert to binary mask (1 for flood, 0 for no flood)
        flood_mask = (predicted == 1).astype(np.uint8) * 255

        return flood_mask

    def save_raster(self, data, output_path, transform, crs):
        # Create a temporary file to store the output
        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
            # Save the data as a GeoTIFF
            with rasterio.open(
                tmp.name,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(data, 1)

            # Copy the temporary file to the final destination
            with open(tmp.name, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())