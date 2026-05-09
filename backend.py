# Suterbrook backend processor

from sentinel1_extractor import FloodMaskModel
from TypedQueue import TypedQueue
import torch


class SuterbrookBackendProcessor:
    def __init__(self, queue):
        self.queue = TypedQueue()
        self.FloodModel = FloodMaskModel()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda_is_available():
            self.total_mem = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.total_mem = 32 * 1024 ** 3 # 32Gb CPU mem

    def clearGPU(self):
        self.FloodModel.unloadModel()

    def GPUMem(self):
        if torch.cuda_is_available():
            stats = torch.cuda.memory_stats(device = self.device)
        else:
            stats = torch.cpu
            return stats["allocated_bytes.all.current"] / self.total_mem

    def process(self):
        while True:
            if self.GPUMem() < 0.8:

                items = self.queue.pop(16)
                if items[0]['type'] == 0:
                    if not self.FloodModel.loaded:
                        self.clearGPU()
                        self.FloodModel.loadModel()
                    print("Processing items: {}".format(items))
                    output = self.FloodModel.forward_batch(items)
