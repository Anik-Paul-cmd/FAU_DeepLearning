from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = None
        TF = tv.transforms
        self.transform_val = TF.Compose([
            TF.ToPILImage(),
            TF.ToTensor(),
            TF.Normalize(mean=train_mean, std=train_std),
        ])

        self.transform_train = TF.Compose([
            TF.ToPILImage(),
            TF.ToTensor(),
            TF.Normalize(mean=train_mean, std=train_std),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        img_path = data_row['filename']

        img_rgb = gray2rgb(imread(img_path, as_gray=True))
        
        label = np.array([data_row['crack'], data_row['inactive']])

        if self.mode == "train":
            img_transformed = self.transform_train(img_rgb)
        elif self.mode == "val":
            img_transformed = self.transform_val(img_rgb)
        
        return img_transformed, label
    