# Suterbrook backend processor

from sentinel1_extractor import FloodMaskModel
from TypedQueue import TypedQueue


class SuterbrookBackendProcessor:
    def __init__(self, queue):
        self.queue = TypedQueue()
        self.FloodModel = FloodMaskModel()

    def process(self):
        while True:
            if FloodModel.GPUMem() < 0.8:
                item = self.queue.pop(16)
            print("Processing item: {}".format(item))
