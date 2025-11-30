import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class NailDataset(Dataset):
    """Dataset for fingernail images."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()

    def _load_dataset(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(root, file)
                    self.image_paths.append(path)
                    if 'anemic' in file.lower() and 'non-anemic' not in file.lower():
                        self.labels.append(1)
                    else:
                        self.labels.append(0)

    def __len__(self) -> int:
        return len(self.image_paths)

    def apply_clahe(self, img_array: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel of LAB image."""
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final_img

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        image = self.apply_clahe(image)
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def get_class_distribution(self):
        """Returns the distribution of classes."""
        dist = {0: 0, 1: 0}
        for label in self.labels:
            dist[label] += 1
        return dist

    def get_class_weights(self):
        """Computes class weights for balanced sampling."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        weights = {0: total / (2 * dist[0]), 1: total / (2 * dist[1])}
        return weights
