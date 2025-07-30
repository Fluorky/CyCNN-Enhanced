import sys

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_loader import CustomIDXDataset


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
    test_images = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    
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

    # train_images = os.path.join(data_dir, 'GTSRB', 'dataset_GTSRB_non_rotated_32x32', 'train-images-idx3-ubyte')
    # train_labels = os.path.join(data_dir, 'GTSRB', 'dataset_GTSRB_non_rotated_32x32', 'train-labels-idx1-ubyte')
    # test_images = os.path.join(data_dir, 'GTSRB', 'dataset_GTSRB_non_rotated_32x32', 'test-images-idx3-ubyte')
    # test_labels = os.path.join(data_dir, 'GTSRB', 'dataset_GTSRB_non_rotated_32x32', 'test-labels-idx1-ubyte')

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


def load_data(dataset='cifar10', data_dir='./data', batch_size=128):

    if dataset == 'cifar10':
        train_set, test_set = load_cifar10_data(data_dir=data_dir, batch_size=batch_size)

    elif  dataset == 'mnist':
        train_set, test_set = load_mnist_data(data_dir=data_dir, batch_size=batch_size)
    
    elif dataset == 'mnist-custom':
        train_set, test_set = load_custom_mnist_data(data_dir=data_dir, batch_size=batch_size)

    elif dataset == 'GTSRB-custom':
        train_set, test_set = load_custom_GTSRB_data(data_dir=data_dir, batch_size=batch_size)
    
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

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    return train_loader, validation_loader, test_loader
