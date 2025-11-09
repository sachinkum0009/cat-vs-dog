"""
Dataset for training
"""

from torchvision.transforms import Compose
from torch.utils.data import Dataset

from PIL import Image

from typing import List

IMAGE_MODE = "RGB"



class CatDogDataset(Dataset):
    def __init__(self, image_paths: List[str], transform: Compose):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self): return self.len

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert(IMAGE_MODE)
        image = self.transform(image)
        label = 0 if "cat" in path else 1
        return (image, label)

