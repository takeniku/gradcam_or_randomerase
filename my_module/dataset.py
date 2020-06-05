from PIL import Image
import os
import torch
from torchvision import transforms, datasets
import numpy as np
from matplotlib import pyplot as plt
device = 'cuda0'

# 画像の前処理を定義


def data_trans(data_dir, cwd):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((585, 414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((585, 414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((585, 414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    return image_datasets


def loaders(batchsize, image_datasets):
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],
                                               batch_size=batchsize,
                                               shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(image_datasets['val'],
                                             batch_size=batchsize,
                                             shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'],
                                              batch_size=batchsize, shuffle=False, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return train_loader, val_loader, test_loader, dataset_sizes, class_names


def imshow(inp, title=None):
    "imshow for Tensor"
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)
