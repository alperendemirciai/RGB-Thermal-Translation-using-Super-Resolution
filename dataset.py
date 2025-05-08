import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchsr.models import edsr

import os
import random
import numpy as np
from PIL import Image
import time

class RGBTDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 split='train',
                 train_ratio=0.8, 
                 sr=False, 
                 random_seed=42):
        """
        Args:
            root_dir (string): Directory with 'color' and 'thermal16' folders
            split (string): 'train', 'val', or 'test'
            train_ratio (float): Ratio of data for training (default: 0.8)
            sr (bool): Apply super resolution to RGB images (default: False)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get file paths
        self.color_dir = os.path.join(root_dir, 'color')
        self.thermal_dir = os.path.join(root_dir, 'thermal16')
        
        # Get all RGB image files
        self.rgb_files = [f for f in os.listdir(self.color_dir) 
                         if os.path.isfile(os.path.join(self.color_dir, f)) 
                         and f.lower().endswith('.jpg')]
        
        # Split dataset
        n_total = len(self.rgb_files)
        n_train = int(n_total * train_ratio)
        n_val_test = n_total - n_train
        n_val = n_val_test // 2
        
        # Generate indices for splits
        indices = list(range(n_total))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Select appropriate indices based on split
        if split == 'train':
            self.indices = train_indices
            self.use_augmentation = True
        elif split == 'val':
            self.indices = val_indices
            self.use_augmentation = False
        elif split == 'test':
            self.indices = test_indices
            self.use_augmentation = False
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Define transformations
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Augmentation transformations for training
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def apply_super_resolution(self, img):
        """Apply super resolution to the input image."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = edsr(scale=2, pretrained=True).to(device)
        model.eval()

        with torch.no_grad():
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            sr_tensor = model(img_tensor).squeeze(0).clamp(0.0, 1.0)
            sr_resized_tensor = torch.nn.functional.interpolate(
                sr_tensor.unsqueeze(0),
                size=img_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            img = transforms.ToPILImage()(sr_resized_tensor.cpu())
        return img  
    
    def get_thermal_path(self, rgb_filename):
        """Convert RGB filename to corresponding thermal filename."""
        # Convert from pattern: "*_eo-*.jpg" to "*_thermal-*.tiff"
        base_name = rgb_filename.replace('_eo-', '_thermal-').rsplit('.', 1)[0]
        thermal_filename = f"{base_name}.tiff"
        thermal_path = os.path.join(self.thermal_dir, thermal_filename)
        
        if not os.path.exists(thermal_path):
            # Try alternative extension
            thermal_filename = f"{base_name}.tif"
            thermal_path = os.path.join(self.thermal_dir, thermal_filename)
            
            if not os.path.exists(thermal_path):
                return None
                
        return thermal_path
    
    def __getitem__(self, idx):
        # Map index to file index
        file_idx = self.indices[idx]
        rgb_filename = self.rgb_files[file_idx]
        
        # Load RGB image
        rgb_path = os.path.join(self.color_dir, rgb_filename)
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Get corresponding thermal image path
        thermal_path = self.get_thermal_path(rgb_filename)
        if thermal_path is None:
            raise FileNotFoundError(f"No matching thermal image found for {rgb_filename}")
        
        thermal_image = Image.open(thermal_path)
        
        # Extract base name for reference
        img_name = os.path.splitext(rgb_filename)[0]
        
        # Apply super resolution if requested (before augmentation)
        if self.sr:
            rgb_image = self.apply_super_resolution(rgb_image)
        
        # Apply consistent augmentations to both images
        if self.use_augmentation:
            # Set a seed to ensure both images get same augmentation
            seed = torch.randint(0, 2147483647, (1,)).item()
            
            torch.manual_seed(seed)
            random.seed(seed)
            rgb_image = self.augmentations(rgb_image)
            
            torch.manual_seed(seed)
            random.seed(seed)
            thermal_image = self.augmentations(thermal_image)
        
        # Apply base transformations
        rgb_tensor = self.base_transform(rgb_image)
        thermal_tensor = self.base_transform(thermal_image)
        
        return {
            'rgb': rgb_tensor, 
            'thermal': thermal_tensor,
            'name': img_name
        }

    @classmethod
    def get_splits(cls, root_dir, train_ratio=0.8, sr=False, random_seed=42):
        """Helper method to get train, val, and test datasets."""
        train_dataset = cls(root_dir, split='train', train_ratio=train_ratio, 
                          sr=sr, random_seed=random_seed)
        val_dataset = cls(root_dir, split='val', train_ratio=train_ratio, 
                        sr=sr, random_seed=random_seed)
        test_dataset = cls(root_dir, split='test', train_ratio=train_ratio, 
                         sr=sr, random_seed=random_seed)
        
        return train_dataset, val_dataset, test_dataset
    

if __name__ == "__main__":
    root_dir = os.path.join(os.getcwd(), 'labeled_rgbt_pairs')  

    st = time.time()
    train_ds, val_ds, test_ds = RGBTDataset.get_splits(root_dir, sr=False)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"First RGB image name: {train_ds[0]['name']}")
    print(f"First thermal image name: {train_ds[0]['name']}")
    print(f"First RGB image shape: {train_ds[0]['rgb'].shape}")
    print(f"First thermal image shape: {train_ds[0]['thermal'].shape}")
    et = time.time()
    print(f"Time taken to load datasets: {et - st:.2f} seconds") # 0.42 seconds

    st = time.time()
    train_ds, val_ds, test_ds = RGBTDataset.get_splits(root_dir, sr=True)
    print(f"Train dataset size with SR: {len(train_ds)}")
    print(f"Validation dataset size with SR: {len(val_ds)}")
    print(f"Test dataset size with SR: {len(test_ds)}")
    print(f"First RGB image name with SR: {train_ds[0]['name']}")
    print(f"First thermal image name with SR: {train_ds[0]['name']}")
    print(f"First RGB image shape with SR: {train_ds[0]['rgb'].shape}")
    print(f"First thermal image shape with SR: {train_ds[0]['thermal'].shape}")
    et = time.time()
    print(f"Time taken to load datasets with SR: {et - st:.2f} seconds") # 48.52 seconds