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


class CustomNPYDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        """
        Dataset class for loading .npy files containing images and labels.

        Args:
            images_path (str): Path to the .npy file with image data.
            labels_path (str): Path to the .npy file with label data.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        # Normalize if needed
        if image.max() > 1:
            image /= 255.0

        # If grayscale: add channel dimension
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Convert HWC to CHW
            image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, int(label)
