# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 17:00
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


from utils.get_parser import get_parser
from torch.utils.data import DataLoader
import torchvision.transforms.functional as ttf
from utils.datasets import *


opt = get_parser()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), ttf.InterpolationMode.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def get_data_loader(mode="train"):
    if mode == "train":
        train_dataloader = DataLoader(
            ImageDataset("datasets/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        return train_dataloader
    else:
        val_dataloader = DataLoader(
            ImageDataset("datasets/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        return val_dataloader
