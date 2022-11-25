import rare_traffic_sign_solution as rtss
import lib

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
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

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

matplotlib.rcParams['figure.figsize'] = (20, 1)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


dataset = rtss.DatasetRTSD(
    ['synthetic_3', 'cropped-train'],
    'classes.json',
)

batch_size = 16

# train_size = int(0.8 * len(dataset))
# valid_size = len(dataset) - train_size
# train_set, valid_set  = random_split(dataset, [train_size, valid_size])
# train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=16, shuffle=False)

train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True)

dataset_test = rtss.TestData('smalltest', 'classes.json', 'smalltest_annotations.csv', return_class_name=False)
test_loader = DataLoader(dataset_test, batch_size=16, num_workers=16)

MyModelCheckpoint = ModelCheckpoint(dirpath='runs/synt_2_finetuning',
                                    filename='{epoch}-{val_acc:.3f}',
                                    monitor='val_acc', 
                                    mode='max', 
                                    save_top_k=1,
                                    save_weights_only=True,
                                    verbose=False)

MyEarlyStopping = EarlyStopping(monitor = "val_acc",
                                mode = "max",
                                patience = 15,
                                verbose = True)

logger = TensorBoardLogger("tb_logs", name="model_synt_finetuned")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    devices=[0],
    callbacks=[MyEarlyStopping, MyModelCheckpoint],
    log_every_n_steps=1,
    enable_progress_bar=False,  
    logger=logger
)

model = rtss.CustomNetwork(features_criterion=None, classes_file='classes.json')
model.load_state_dict(torch.load('/home/alexkkir/cv-hw10-road-signs/runs/synt_2/epoch=13-val_acc=0.978.ckpt', map_location='cpu')['state_dict'])
trainer.fit(model, train_loader, test_loader)