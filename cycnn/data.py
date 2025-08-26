import sys

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_loader import CustomIDXDataset, CustomNPYDataset


def load_mnist_data(data_dir='./data', batch_size=128):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    return train_set, test_set

def load_cifar10_data(data_dir='./data', batch_size=128):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_set, test_set


# TODO refator these three methods and merge them into single method

def load_custom_mnist_data(data_dir='./data', batch_size=128):
    import os

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        
    train_images = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images = os.path.join(data_dir, 'test-images-idx3-ubyte')
    test_labels = os.path.join(data_dir, 'test-labels-idx1-ubyte')
    
    train_set = CustomIDXDataset(images_path=train_images, labels_path=train_labels, transform=train_transform)
    test_set = CustomIDXDataset(images_path=test_images, labels_path=test_labels, transform=test_transform)

    return train_set, test_set

def load_custom_GTSRB_data(data_dir='./data', batch_size=128):
    import os

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_images = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images = os.path.join(data_dir, 'test-images-idx3-ubyte')
    test_labels = os.path.join(data_dir, 'test-labels-idx1-ubyte')

    train_set = CustomIDXDataset(images_path=train_images, labels_path=train_labels, transform=train_transform)
    test_set = CustomIDXDataset(images_path=test_images, labels_path=test_labels, transform=test_transform)

    return train_set, test_set



def load_LEGO_data(data_dir='./data', batch_size=128):
    import os

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_images = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images = os.path.join(data_dir, 'test-images-idx3-ubyte')
    test_labels = os.path.join(data_dir, 'test-labels-idx1-ubyte')

    train_set = CustomIDXDataset(images_path=train_images, labels_path=train_labels, transform=train_transform)
    test_set =CustomIDXDataset(images_path=test_images, labels_path=test_labels, transform=test_transform)

    return train_set, test_set

def load_cifar100_data(data_dir='./data', batch_size=128):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    return train_set, test_set


def load_svhn_data(data_dir='./data', batch_size=128):


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))
    ])

    train_set = torchvision.datasets.SVHN(root=data_dir, split='train', transform=train_transform, download=True)
    test_set = torchvision.datasets.SVHN(root=data_dir, split='test', transform=test_transform, download=True)

    return train_set, test_set

def load_custom_GTSRB_RGB_data(data_dir='./data', batch_size=128):
    import os

    # Stats for the GTSRB RGB
    normalize_mean = (0.3403, 0.3121, 0.3214)
    normalize_std = (0.2724, 0.2608, 0.2669)

    transform = transforms.Compose([
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    train_set = CustomNPYDataset(
        images_path=os.path.join(data_dir, 'train_images.npy'),
        labels_path=os.path.join(data_dir, 'train_labels.npy'),
        transform=transform
    )
    test_set = CustomNPYDataset(
        images_path=os.path.join(data_dir, 'test_images.npy'),
        labels_path=os.path.join(data_dir, 'test_labels.npy'),
        transform=transform
    )

    return train_set, test_set

def load_data(dataset='cifar10', data_dir='./data', batch_size=128):

    if dataset == 'cifar10':
        train_set, test_set = load_cifar10_data(data_dir=data_dir, batch_size=batch_size)

    elif  dataset == 'mnist':
        train_set, test_set = load_mnist_data(data_dir=data_dir, batch_size=batch_size)
    
    elif dataset == 'mnist-custom':
        train_set, test_set = load_custom_mnist_data(data_dir=data_dir, batch_size=batch_size)

    elif dataset == 'GTSRB-custom':
        train_set, test_set = load_custom_GTSRB_data(data_dir=data_dir, batch_size=batch_size)
    
    elif dataset == 'GTSRB-RGB-custom':
        train_set, test_set = load_custom_GTSRB_RGB_data(data_dir=data_dir, batch_size=batch_size)
    
    elif dataset == 'LEGO':
        train_set, test_set = load_LEGO_data(data_dir=data_dir, batch_size=batch_size)

    elif  dataset == 'cifar100':
        train_set, test_set = load_cifar100_data(data_dir=data_dir, batch_size=batch_size)

    elif  dataset == 'svhn':
        train_set, test_set = load_svhn_data(data_dir=data_dir, batch_size=batch_size)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    """Give 10% of the train set to validation set"""
    train_set, validation_set = torch.utils.data.random_split(train_set, [len(train_set) - len(train_set) // 10, len(train_set) // 10])

    _num_workers = 8  
    _pin_memory = True
    _persistent_workers = True if _num_workers > 0 else False
    _prefetch_factor = 4 if _num_workers > 0 else None

    common_kwargs = dict(
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        # persistent_workers=_persistent_workers,
        # prefetch_factor=_prefetch_factor,
    )
   
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True, 
                                               **common_kwargs)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    **common_kwargs)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              **common_kwargs)

    return train_loader, validation_loader, test_loader
