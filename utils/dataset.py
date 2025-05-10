from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, List
from torchvision.transforms import functional as F


class RGBT_Dataset(Dataset):
    def __init__(self, data_dir: str, transform: transforms.Compose = None, random_state: int = 42, sr:bool = False, 
                 thermal_type: str = "thermal8", train_ratio:float = 0.8, val_ratio:float = 0.1, mode:str = "train"):
        
        self.data_dir = data_dir
        self.transform = transform
        self.random_state = random_state
        self.sr = sr
        self.thermal_type = thermal_type

        self.thermal_directory = os.path.join(data_dir, thermal_type)
        self.rgb_directory = os.path.join(data_dir, "color")
        self.sr_directory = os.path.join(data_dir, "sr")

        self.mode = mode
        if self.mode not in ["train", "val", "test"]:
            raise ValueError("mode must be one of ['train', 'val', 'test']")

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        train_indexes = []
        val_indexes = []
        test_indexes = []

        all_indexes = list(range(len(os.listdir(self.thermal_directory))))

        random.shuffle(all_indexes)
        num_train = int(len(all_indexes) * train_ratio)
        num_val = int(len(all_indexes) * val_ratio)
        num_test = len(all_indexes) - num_train - num_val

        train_indexes = all_indexes[:num_train]
        val_indexes = all_indexes[num_train:num_train + num_val]
        test_indexes = all_indexes[num_train + num_val:]

        if self.mode == "train":
            self.indexes = train_indexes
        elif self.mode == "val":
            self.indexes = val_indexes
        else:
            self.indexes = test_indexes

        self.thermal_paths = sorted(os.listdir(self.thermal_directory))
        self.rgb_paths = sorted(os.listdir(self.rgb_directory))
        self.sr_paths = sorted(os.listdir(self.sr_directory)) if sr else None

    def __len__(self) -> int:
        return len(self.indexes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        thermal_file_name = self.thermal_paths[self.indexes[idx]]
        thermal_path = os.path.join(self.thermal_directory, thermal_file_name)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if self.thermal_type == "thermal8":    
            thermal = np.clip(thermal, 0, 255)
            thermal = (thermal / 255.0).astype(np.float32)
        elif self.thermal_type == "thermal16":
            thermal = np.clip(thermal, 0, 65535)
            thermal = (thermal / 65535.0).astype(np.float32)

        
        if self.sr:
            color_file_name = self.sr_paths[self.indexes[idx]]
            color_path = os.path.join(self.sr_directory, color_file_name)
        else:
            color_file_name = self.rgb_paths[self.indexes[idx]]
            color_path = os.path.join(self.rgb_directory, color_file_name)

        color = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.clip(color, 0, 255)
        color = (color / 255.0).astype(np.float32)
        

        if self.transform:
            thermal = np.reshape(thermal, (1, thermal.shape[0], thermal.shape[1]))
            color = np.transpose(color, (2, 0, 1))
            thermal = np.transpose(thermal, (0, 1, 2))

            cascaded= np.concatenate((color, thermal), axis=0)
            cascaded = np.transpose(cascaded, (1, 2, 0))
            cascaded = self.transform(cascaded)
            color = cascaded[:3, :, :]
            thermal = cascaded[3:, :, :]
        else:
            color = transforms.ToTensor()(color)
            thermal = transforms.ToTensor()(thermal)

        return color, thermal
    

if __name__ == "__main__":
    
    tforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    dataset = RGBT_Dataset(data_dir="../labeled_rgbt_pairs", sr=False, thermal_type="thermal8", mode="train", transform=tforms)
    color, thermal = dataset[7]


    color = np.transpose(color.numpy(), (1, 2, 0))
    thermal = np.transpose(thermal.numpy(), (1, 2, 0))

    from matplotlib import pyplot as plt

    plt.imshow(color)
    plt.title("Color Image")
    plt.axis("off")
    plt.show()
    plt.imshow(thermal, cmap="gray")
    plt.title("Thermal Image")
    plt.axis("off")
    plt.show()

    print(color.shape)
    print(thermal.shape)

    

        
