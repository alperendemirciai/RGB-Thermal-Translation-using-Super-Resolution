import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum value of the image (default: 1.0)
        
    Returns:
        psnr_val: PSNR value
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        data_range: Range of the image data (default: 1.0)
        
    Returns:
        ssim_val: SSIM value
    """
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    return ssim(img1_np, img2_np, data_range=data_range, multichannel=False)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image

def save_checkpoint(model, optimizer, epoch, filename):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        filename: File path for saving checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filepath, model, optimizer=None, lr=None, device=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load parameters into
        optimizer: PyTorch optimizer to load state into (optional)
        lr: Learning rate to set (optional)
        device: Device to load the model on (optional)
        
    Returns:
        epoch: The epoch number of the loaded checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return 0
        
    if device is None:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    return checkpoint['epoch']


def save_image(img_array, filepath):
    """
    Save a NumPy image array to a file
    
    Args:
        img_array: NumPy array of shape [H, W, C] with values in [0, 1]
        filepath: Path to save the image
    """
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    # Handle different channel numbers
    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array[:, :, 0], 'L')
    elif img_array.shape[2] == 3:
        img = Image.fromarray(img_array, 'RGB')
    elif img_array.shape[2] == 4:
        img = Image.fromarray(img_array, 'RGBA')
    else:
        # For other channel numbers, save first three channels as RGB
        img = Image.fromarray(img_array[:, :, :3], 'RGB')
    
    img.save(filepath)

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum value of the image (default: 1.0)
        
    Returns:
        psnr_val: PSNR value
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def save_some_examples(gen, val_loader, epoch, folder, device, num_samples=4, denorm=False):
    os.makedirs(folder, exist_ok=True)
    
    x, y = next(iter(val_loader))
    x, y = x[:num_samples].to(device), y[:num_samples].to(device)
    
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
    gen.train()

    for i in range(num_samples):
        input_img = x[i].cpu()
        target_img = y[i].cpu()
        pred_img = y_fake[i].cpu()

        if denorm:
            input_img = (input_img + 1) / 2
            target_img = (target_img + 1) / 2
            pred_img = (pred_img + 1) / 2

        input_img = TF.to_pil_image(input_img)
        target_img = TF.to_pil_image(target_img)
        pred_img = TF.to_pil_image(pred_img)

        # Create subfolder for each triplet
        triplet_folder = os.path.join(folder, f"img{i+1}_epoch{epoch}")
        os.makedirs(triplet_folder, exist_ok=True)

        input_img.save(os.path.join(triplet_folder, "input.png"))
        target_img.save(os.path.join(triplet_folder, "target.png"))
        pred_img.save(os.path.join(triplet_folder, "prediction.png"))


def plot_metrics(train_metric, val_metric, metric_name, folder, epoch):
    """
    Plot training and validation metrics
    
    Args:
        train_metric: Training metric values
        val_metric: Validation metric values
        metric_name: Name of the metric (e.g., 'PSNR', 'SSIM')
        folder: Folder to save the plot
        epoch: Current epoch number
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_metric, label='Train', color='blue')
    plt.plot(val_metric, label='Validation', color='orange')
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    
    # Save the plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{metric_name}_epoch_{epoch}.png"))
    plt.close()

def plot_metric(metric, metric_name, folder, epoch):
    """
    Plot a single metric over epochs
    
    Args:
        metric: Metric values
        metric_name: Name of the metric (e.g., 'PSNR', 'SSIM')
        folder: Folder to save the plot
        epoch: Current epoch number
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric, label=metric_name, color='blue')
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    
    # Save the plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{metric_name}_epoch_{epoch}.png"))
    plt.close()