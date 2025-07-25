import torch
from torch.utils.data import Dataset
import numpy as np
import os
import struct

class CustomIDXDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)

    def _read_images(self, path):
        with open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            images = image_data.reshape(num, rows, cols).copy()
        return images

    def _read_labels(self, path):
        with open(path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            label_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return label_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.unsqueeze(0)  # Add channel dim
        return image, label


class CustomGTSRBDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)

    def _read_images(self, path):
        with open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            images = image_data.reshape(num, rows, cols).copy()
        return images

    def _read_labels(self, path):
        with open(path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            label_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return label_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.unsqueeze(0)  # Add channel dim
        return image, label