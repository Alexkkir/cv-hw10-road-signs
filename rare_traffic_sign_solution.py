# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from albumentations.augmentations.geometric import functional as AF
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import os
import pandas as pd
import json
import tqdm
import pickle
import typing
from collections import defaultdict
import random

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

from albumentations.core.transforms_interface import DualTransform, BasicTransform


CLASSES_CNT = 205

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(self, root_folders, path_to_classes_json, use_augmentations=True) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(
            path_to_classes_json)
        self.use_augmentations = use_augmentations
        # YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.samples = []
        # YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = defaultdict(list)

        for root_folder in root_folders:
            index = 0
            for folder in os.listdir(root_folder):
                for name in os.listdir(root_folder + '/' + folder):
                    path = root_folder + '/' + folder + '/' + name
                    self.samples.append((path, self.class_to_idx[folder]))
                    self.classes_to_samples[self.class_to_idx[folder]].append(
                        index)
                    index += 1

        for cls in self.class_to_idx.values():
            if cls not in self.classes_to_samples:
                self.classes_to_samples[cls] = []
        self.classes_to_samples = dict(self.classes_to_samples)

        # YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.transform = None
        if self.use_augmentations:
            self.transform = A.Compose([
                A.FromFloat('uint8'),
                # A.Resize(224, 224, always_apply=True),
                A.Resize(256, 256, always_apply=True),
                A.RandomCrop(224, 224, always_apply=True),
                # A.HorizontalFlip(p=0.3),
                A.Rotate(p=0.35, limit=15),
                A.RingingOvershoot(p=0.2, blur_limit=(3, 7)),
                A.OneOf([
                    A.HueSaturationValue(p=0.5),
                    A.RGBShift(p=0.3),
                    A.Compose([
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                        A.CLAHE(p=0.5),
                    ], p=1)
                ], p=0.5),
                A.Affine(scale=(0.85, 1), translate_percent=(
                    0, 0.10), shear=(-4, 4), p=0.35),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.FromFloat('uint8'),
                # A.Resize(224, 224, always_apply=True),
                A.Resize(256, 256, always_apply=True),
                A.RandomCrop(224, 224, always_apply=True),
                # A.HorizontalFlip(p=0.3),
                # A.Rotate(p=0.35, limit=15),
                # A.RingingOvershoot(p=0.2, blur_limit=(3, 7)),
                # A.OneOf([
                #     A.HueSaturationValue(p=0.5),
                #     A.RGBShift(p=0.3),
                #     A.Compose([
                #         A.RandomBrightnessContrast(p=0.5),
                #         A.RandomGamma(p=0.5),
                #         A.CLAHE(p=0.5),
                #     ], p=1)
                # ], p=0.5),
                # A.Affine(scale=(0.85, 1), translate_percent=(
                #     0, 0.10), shear=(-4, 4), p=0.35),
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        # YOUR CODE HERE
        sample = self.samples[index]
        image = plt.imread(sample[0])
        if image.dtype == 'uint8':
            image = image.astype('float') / 255
        image = self.transform(image=image)['image']
        return image, *sample

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json) as f:
            classes_json = json.load(f)

         # YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        class_to_idx = {k: v['id'] for k, v in classes_json.items()}
        # YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = [(v, k) for k, v in class_to_idx.items()]
        classes = sorted(classes, key=lambda x: x[0])
        classes = [x[1] for x in classes]
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(self, root, path_to_classes_json, annotations_file=None, return_class_name=True):
        super(TestData, self).__init__()
        self.root = root
        self.return_class_name = return_class_name
        self.samples = []  # YOUR CODE HERE - список путей до картинок

        with open(path_to_classes_json) as f:
            classes_json = json.load(f)

        for file in os.listdir(root):
            path = file
            self.samples.append(path)

        self.transform = A.Compose([
            A.FromFloat('uint8'),
            # A.Resize(224, 224, always_apply=True),
            A.Resize(256, 256, always_apply=True),
            A.CenterCrop(224, 224, always_apply=True),
            A.Normalize(),
            ToTensorV2(),
        ])

        self.targets = None
        if annotations_file is not None:
            # YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            self.targets = defaultdict(lambda: -1)
            annotations = pd.read_csv(annotations_file)
            annotations = dict(
                zip(annotations['filename'], annotations['class']))
            for file in self.samples:
                if file in annotations:
                    self.targets[file] = classes_json[annotations[file]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        # YOUR CODE HERE
        sample = self.samples[index]
        file = sample
        image = plt.imread(self.root + '/' + file)
        if image.dtype == 'uint8':
            image = image.astype('float') / 255
        image = self.transform(image=image)['image']
        if self.return_class_name:
            return image, file, -1 if self.targets is None else self.targets[file]
        else:
            return image, file, -1 if self.targets is None else self.targets[file]['id']


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(self, features_criterion=None, internal_features=1024, classes_file=None):
        super(CustomNetwork, self).__init__()

        features = list(torchvision.models.resnet50(
            # weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).children())[:-2]
        features = nn.Sequential(*features)
        self.features = features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, 205)
        )

        self.loss = nn.CrossEntropyLoss()
        self.acc = lambda pred, y: torch.sum(
            pred.argmax(dim=1) == y) / y.shape[0]

        self.class_name_to_type = None
        if classes_file is not None:
            with open(classes_file, "r") as fr:
                classes_info = json.load(fr)
            self.class_name_to_type = {v['id']: v['type'] for k, v in classes_info.items()}

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        x = self.forward(x)
        x = F.softmax(x, dim=1).detach().cpu().argmax(dim=1).numpy()
        return x  # YOUR CODE HERE

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.argmax(dim=1)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch[0], batch[2]

        pred = self(x)
        loss = self.loss(pred, y)
        acc = self.acc(pred, y)

        y_true = list(int(t) for t in y)
        y_pred = list(int(t) for t in pred.argmax(dim=1))

        return {'loss': loss, 'acc': acc, 'y_true': y_true, 'y_pred': y_pred}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.features.parameters(), 'lr': 3e-5},
            {'params': self.classifier.parameters()}
        ], lr=3e-4, weight_decay=3e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.2,
            patience=5,
            verbose=True)

        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_acc"
        }

        return [optimizer], [lr_dict]

    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch[0], batch[2]
        pred = self(x)
        loss = self.loss(pred, y)
        acc = self.acc(pred, y)

        y_true = list(int(t) for t in y)
        y_pred = list(int(t) for t in pred.argmax(dim=1))

        return {'val_loss': loss, 'val_acc': acc, 'y_true': y_true, 'y_pred': y_pred}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        y_true = sum([x['y_true'] for x in outputs], [])
        y_pred = sum([x['y_pred'] for x in outputs], [])

        total_acc = calc_metric(y_true, y_pred, 'all',self.class_name_to_type)
        rare_recall = calc_metric(y_true, y_pred, 'rare', self.class_name_to_type)
        freq_recall = calc_metric(y_true, y_pred, 'freq', self.class_name_to_type)

        print(f"| TRAIN acc: {avg_acc:.2f}, loss: {avg_loss:.2f}, total_acc: {total_acc:.2f}, rare_recall: {rare_recall:.2f}, freq_recall: {freq_recall:.2f}")

        self.log('train_loss', avg_loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('train_acc', avg_acc, prog_bar=True,
                 on_epoch=True, on_step=False)

    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        y_true = sum([x['y_true'] for x in outputs], [])
        y_pred = sum([x['y_pred'] for x in outputs], [])

        total_acc = calc_metric(y_true, y_pred, 'all',self.class_name_to_type)
        rare_recall = calc_metric(y_true, y_pred, 'rare', self.class_name_to_type)
        freq_recall = calc_metric(y_true, y_pred, 'freq', self.class_name_to_type)

        print(
            f"[Epoch {self.trainer.current_epoch:3}] VALID acc: {avg_acc:.2f}, loss: {avg_loss:.2f}, total_acc: {total_acc:.2f}, rare_recall: {rare_recall:.2f}, freq_recall: {freq_recall:.2f}", end=" ")

        self.log('val_loss', avg_loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_acc', avg_acc, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_total_acc', total_acc, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_rare_recall', rare_recall, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_freq_recall', freq_recall, prog_bar=True,
                 on_epoch=True, on_step=False)        


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    # YOUR CODE HERE
    dataset = DatasetRTSD(
        ['cropped-train'],
        'classes.json',
    )

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set  = random_split(dataset, [train_size, valid_size])

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=16, shuffle=False)

    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/synt_2',
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

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=[0],
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        log_every_n_steps=1,
        enable_progress_bar=False,
    )

    model = CustomNetwork(classes_file='classes.json')
    trainer.fit(model, train_loader, valid_loader)

    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    with open(path_to_classes_json) as f:
        classes_json = json.load(f)

    # YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
    class_to_idx = {k: v['id'] for k, v in classes_json.items()}
    # YOUR CODE HERE - массив, classes[индекс] = 'название класса'
    classes = [(v, k) for k, v in class_to_idx.items()]
    classes = sorted(classes, key=lambda x: x[0])
    classes = [x[1] for x in classes]

    dataset = TestData(test_folder, path_to_classes_json)
    batch_size = 4
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=batch_size)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    out = []
    for batch in tqdm.tqdm(loader):
        images = batch[0]
        images = images.to(device)
        pred = model.predict(images)
        for i in range(len(batch[1])):
            name = batch[1][i]
            label_idx = int(pred[i])
            label = classes[label_idx]
            out.append({'filename': name, 'class': label})
    # YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    return out


# def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
#     """
#     Функция для тестирования качества модели.
#     Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
#     :param model: модель, которую нужно протестировать
#     :param test_folder: путь до папки с тестовыми данными
#     :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
#     """
#     # YOUR CODE HERE
#     dataset = TestData(test_folder, path_to_classes_json, annotations_file)
#     return total_acc, rare_recall, freq_recall

class RandomResize:
    def __init__(self, limit, p=1):
        self.limit = limit
        self.p = p
    
    def __call__(self, img):
        new_size = np.random.randint(*self.limit)
        return cv2.resize(img, (new_size, new_size))
        

class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path):
        # YOUR CODE HERE
        self.background_path = background_path

        samples = []
        for file in os.listdir(background_path):
            path = Path(background_path) / file
            path = str(path)
            samples.append(path)
        self.samples = samples

        self.transforms_0 = RandomResize((64, 128))

        self.transforms_1 = A.Compose([
            A.FromFloat(dtype='uint8'),
            A.RandomBrightnessContrast(p=1),
            A.RGBShift(p=1),
            A.ToFloat()
        ])

        self.transforms_2 = A.Compose([
            # A.Resize(64, 64),
            A.CropAndPad(percent=(0, 0.15)),
            A.Rotate((15), border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
            A.MotionBlur((15), p=1),
            A.Blur((5, 15), p=1),
        ])

        self.contrast_factor = 0.7

    def _merge(self, background, icon):
        n, m = background.shape[:2]
        k, l = icon.shape[:2]
        
        a, b = np.random.randint(n - k), np.random.randint(m - l)
        zone = background[a:a+k, b:b+l]
        icon, mask = icon[..., 0:3], icon[..., 3:4]
        mask /= mask.max()
        zone= zone * (1 - mask) + icon * mask
        return zone

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        bg = random.choice(self.samples)
        bg = plt.imread(bg) / 255

        icon = self.transforms_0(icon) * self.contrast_factor
        icon_t = icon.copy()
        icon_t[..., 0:3] = self.transforms_1(image=icon[..., 0:3])['image']
        icon_t = self.transforms_2(image=icon_t)['image']

        icon_t = self._merge(bg, icon_t)
        icon_t = np.clip(icon_t, 0, 1)
                
        return icon_t


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    path_icon, path_dst, path_backgrounds, n_samples = args
    path_icon, path_dst, path_backgrounds = Path(path_icon), Path(path_dst), Path(path_backgrounds)
    class_name = str(path_icon.name).replace('.png', '')

    (path_dst / class_name).mkdir(exist_ok=True)
    
    sg = SignGenerator(path_backgrounds)
    icon = plt.imread(path_icon)
    for i in range(n_samples):
        new_icon = sg.get_sample(icon)
        plt.imsave(path_dst / class_name / ('%04d.jpg' % (i,)), new_icon)

def generate_all_data(output_folder, icons_path, background_path, samples_per_class=1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    Path(output_folder).mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params), total=len(params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    # YOUR CODE HERE
    dataset = DatasetRTSD(
        ['synthetic_3', 'cropped-train'],
        'classes.json',
    )

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set  = random_split(dataset, [train_size, valid_size])

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=16, shuffle=False)

    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/synt_2',
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

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=[0],
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        log_every_n_steps=1,
        enable_progress_bar=False,
    )

    model = CustomNetwork(classes_file='classes.json')
    trainer.fit(model, train_loader, valid_loader)

    return model


class ArcLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcLoss, self).__init__()
        self.s = 30.0
        self.m = 0.4
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = 1e-7

    def forward(self, x, labels):
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        cos_theta = self.fc(x)
        numerator = torch.diagonal(cos_theta.transpose(0, 1)[labels]) # возьмем диагональные элементы из cos_theta.transpose(0, 1)[labels]
        numerator = torch.clamp(numerator, -1 + self.eps, 1 - self.eps)
        numerator = self.s * torch.cos(torch.acos(numerator) + self.m)
        
        excl = torch.cat([torch.cat((cos_theta[i, :y], cos_theta[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return cos_theta, -torch.mean(L)



class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(self, data_source: DatasetRTSD, elems_per_class: int, classes_per_batch: int, max_iters=None):
        # YOUR CODE HERE
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.iter = 0
        self.max_iters = max_iters or len(data_source) / (classes_per_batch * elems_per_class)
        # self.max_iters = 2


    def __iter__(self):
        # YOUR CODE HERE
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.max_iters:
            self.iter += 1
            classes = random.sample(list(self.data_source.classes_to_samples), self.classes_per_batch)
            indices = []
            for cls in classes:
                indices += random.choices(self.data_source.classes_to_samples[cls], k=self.elems_per_class)
            return indices
        else:
            raise StopIteration


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
   
    dataset = DatasetRTSD(
        ['synthetic_3', 'cropped-train'],
        'classes.json',
    )

    classes_per_batch = 32
    elems_per_class = 4

    sampler_1 = CustomBatchSampler(dataset, classes_per_batch=classes_per_batch, elems_per_class=elems_per_class)
    train_loader = DataLoader(dataset, batch_sampler=sampler_1, num_workers=16)

    # model
    name = "features_new"
    MyModelCheckpoint = ModelCheckpoint(dirpath=f'runs/{name}',
                                        filename='{epoch}-{train_loss:.3f}',
                                        monitor='train_loss', 
                                        mode='min', 
                                        save_top_k=1,
                                        save_weights_only=True,
                                        verbose=False)

    MyEarlyStopping = EarlyStopping(monitor = "train_loss",
                                    mode = "min",
                                    patience = 15,
                                    verbose = True)

    logger = TensorBoardLogger("tb_logs", name=name)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=[0],
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=logger
    )

    model = ModelWithHead()
    trainer.fit(model, train_loader)

    return model


class ModelWithHead(pl.LightningModule):
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors=4):
        super().__init__()
        # YOUR CODE HERE
        self.backbone = torchvision.models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.backbone.fc = nn.Linear(2048, 1024)
        self.classifier = ArcLoss(1024, 205)

        self.knn: KNeighborsClassifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='euclidean'
        )

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        # YOUR CODE HERE
        self.load_state_dict(torch.load(nn_weights_path)['state_dict'])

    def load_head(self, knn_path):
        """ 
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        # YOUR CODE HERE
        with open(knn_path, 'rb') as f:
            X_train = pickle.load(f)
            Y_train = pickle.load(f)
        self.knn.fit(X_train, Y_train)

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        # YOUR CODE HERE - предсказание нейросетевой модели
        x = self.forward(imgs)
        x = x.detach().cpu().numpy()
        x = x.reshape(x.shape[0], -1)
        x = x / np.linalg.norm(x, axis=1)[:, None]
        knn_pred = self.knn.predict(x)  # YOUR CODE HERE - предсказание kNN на features
        return knn_pred

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[2]
        outputs = self.backbone(x)
        outputs, loss = self.classifier(outputs, y)
        acc = (outputs.argmax(dim=1) == y).sum().item() / len(outputs)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[2]
        outputs = self.backbone(x)
        outputs, loss = self.classifier(outputs, y)
        acc = (outputs.argmax(dim=1) == y).sum().item() / len(outputs)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 3e-5},
        ], lr=3e-4, weight_decay=3e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            verbose=True)

        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train_loss"
        }

        return [optimizer], [lr_dict]
    
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print(f"| TRAIN loss: {avg_loss:.2f}")

        self.log('train_loss', avg_loss, prog_bar=True,
                 on_epoch=True, on_step=False)


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class) -> None:
        # YOUR CODE HERE
        self.data_source = data_source
        self.examples_per_class = examples_per_class
        self.iter = 0
        self.max_iters = len(data_source.classes) * examples_per_class

    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        self.iter = 0
        chosen_indices = []
        for class_indices in self.data_source.classes_to_samples.values():
            chosen_indices += random.choices(class_indices, k=self.examples_per_class)
        self.chosen_indices = chosen_indices        
        return self # YOUR CODE HERE

    def __next__(self):
        if self.iter < self.max_iters:
            i = self.iter
            self.iter += 1
            return self.chosen_indices[i]
        else:
            raise StopIteration
                


def train_head(nn_weights_path, examples_per_class=20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    # YOUR CODE HERE
