# dataset.py

import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from os.path import join
from os import listdir

LR_MIN = 0.0206
LR_MAX = 0.2737
HR_MIN = 0.0191
HR_MAX = 0.4274

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.TIF', '.tif'])

class Oli2MSIDataset(Dataset):
    def __init__(self, lr_files, hr_files, hr_crop_size=480, upscale_factor=4, is_train=True):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // upscale_factor
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.is_train = is_train

        assert len(self.lr_files) == len(self.hr_files), "Mismatch in number of LR and HR files"

    def __getitem__(self, index):
        lr_file = self.lr_files[index]
        hr_file = self.hr_files[index]

        with rasterio.open(lr_file) as src:
            lr_image = src.read().astype(np.float32)
        with rasterio.open(hr_file) as src:
            hr_image = src.read().astype(np.float32)

        lr_image = np.clip((lr_image - LR_MIN) / (LR_MAX - LR_MIN), 0.0, 1.0)
        hr_image = np.clip((hr_image - HR_MIN) / (HR_MAX - HR_MIN), 0.0, 1.0)

        if self.is_train:
            lr_h, lr_w = lr_image.shape[1], lr_image.shape[2]
            rand_x_lr = random.randint(0, lr_w - self.lr_crop_size)
            rand_y_lr = random.randint(0, lr_h - self.lr_crop_size)
            lr_cropped = lr_image[:, rand_y_lr:rand_y_lr + self.lr_crop_size, rand_x_lr:rand_x_lr + self.lr_crop_size]

            rand_x_hr = rand_x_lr * self.upscale_factor
            rand_y_hr = rand_y_lr * self.upscale_factor
            hr_cropped = hr_image[:, rand_y_hr:rand_y_hr + self.hr_crop_size, rand_x_hr:rand_x_hr + self.hr_crop_size]

            if random.random() > 0.5:
                lr_cropped = np.ascontiguousarray(np.flip(lr_cropped, axis=2))
                hr_cropped = np.ascontiguousarray(np.flip(hr_cropped, axis=2))
            if random.random() > 0.5:
                lr_cropped = np.ascontiguousarray(np.flip(lr_cropped, axis=1))
                hr_cropped = np.ascontiguousarray(np.flip(hr_cropped, axis=1))

            return torch.from_numpy(lr_cropped), torch.from_numpy(hr_cropped)
        else:
            return torch.from_numpy(lr_image), torch.from_numpy(hr_image)

    def __len__(self):
        return len(self.lr_files)
