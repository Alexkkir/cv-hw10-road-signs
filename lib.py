import rare_traffic_sign_solution as rtss
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import floor, ceil
from sklearn.model_selection import train_test_split

import shutil
import requests
import functools
import pathlib
from pathlib import Path
import shutil
from tqdm.notebook import tqdm
import os
from collections import defaultdict

from IPython.display import clear_output

matplotlib.rcParams['figure.figsize'] = (20, 5)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def display_loader(loader, cols=8):
    batch = next(iter(loader))
    rows = ceil(len(batch[0]) / cols)
    fig_size = matplotlib.rcParams['figure.figsize'][0] / cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize = (fig_size * cols, fig_size * rows))
    for i, image in enumerate(batch[0]):
        image = image.permute(1, 2, 0)
        image = image * STD + MEAN
        label = int(batch[2][i])
        color = 'black'
        ax.ravel()[i].imshow((image * 255).type(torch.uint8))
        ax.ravel()[i].set_title(label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def modify_dataset(root, new_root, image_size):
    root = Path(root)
    new_root = Path(new_root)
    os.mkdir(new_root)
    for folder in tqdm(os.listdir(root)):
        os.mkdir(new_root / folder)
        for file in os.listdir(root / folder):
            image = cv2.imread(str(root / folder / file))
            image = cv2.resize(image, image_size)
            cv2.imwrite(str(new_root / folder / file), image)

def test_on_dataloader(model, loader, repeats=3, device=1):
    device = torch.device(f'cuda:{device}')
    model.to(device)
    model.eval()
    correct = defaultdict()
    predicted = defaultdict(list)
    for repeat in range(repeats):
        for batch in loader:
            images, names, labels = batch
            for name, label in zip(names, labels):
                correct[name] = label

            images = images.to(device)
            pred = model.forward(images).detach().cpu()
            pred = F.softmax(pred, dim=1).argmax(dim=1)

            for name, p in zip(names, pred):
                predicted[name].append(p)

    predicted_most_common = {}
    for name, preds in predicted.items():
        most_common = max(set(preds), key=preds.count)
        predicted_most_common[name] = most_common

    n = 0
    for k in correct:
        true = correct[k]
        pred = predicted_most_common[k]
        n += true == pred
    n = float(n) / len(correct)

    acc_mean = n

    acc_single = []
    for i in range(repeats):
        n = 0
        for k in correct:
            true = correct[k]
            pred = predicted[k][i]
            n += true == pred
        n = float(n) / len(correct)

        acc_single.append(n)
    return acc_mean, acc_single