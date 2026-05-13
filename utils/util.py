import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import PIL

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class EditingJsonDataset(Dataset):

    def __init__(self, args, repeats=1):
        self.image_dir = args.image_dir_path
        self.transform = build_transform(args)

        with open(args.json_file, "r") as f:
            self.image_prompt = json.load(f)

        self.image_files = list(self.image_prompt.keys()) * repeats

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        original_prompt, editing_prompt = self.image_prompt[self.image_files[idx]]

        if self.transform is not None:
            image = self.transform(image)

        return image, original_prompt, editing_prompt


class EditingSingleImageDataset(Dataset):

    def __init__(self, args, repeats=1):
        self.transform = build_transform(args)
        self.image_files = [args.image_file_path] * repeats
        self.original_prompt = args.image_caption
        self.editing_prompt = args.editing_prompt

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.original_prompt, self.editing_prompt


def build_transform(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if args.image_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0

    resize_size = int(args.image_size / crop_pct)

    transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=PIL.Image.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
