# Suterbrook backend processor

import torch
from fastapi import fastapi

from sentinel1_extractor import FloodMaskModel
from TypedQueue import TypedQueue
from utils import *


class SuterbrookBackendProcessor:
    def __init__(self):
        self.FloodModelQueue = TypedQueue()
        self.FloodModel = FloodMaskModel()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.total_mem = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.total_mem = 32 * 1024**3  # 32Gb CPU mem
        return None

    def clearGPU(self):
        self.FloodModel.unloadModel()
        return None

    def GPUMem(self):
        if torch.cuda_is_available():
            stats = torch.cuda.memory_stats(device=self.device)
            return stats["allocated_bytes.all.current"] / self.total_mem
        else:
            return 1.0

    def process(self):
        while self.running:
            if self.GPUMem() < 0.8:
                items = self.queue.pop(16)
                if items[0]["type"] == 0:
                    if not self.FloodModel.loaded:
                        self.clearGPU()
                        self.FloodModel.loadModel()
                    print("Processing items: {}".format(items))
                    output = [x.cpu() for x in self.FloodModel.forward_batch(items)]

        return None

    def shutdown(self):
        self.running = False
        self.executor.shutdown(wait=True)
