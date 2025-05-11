import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

from utils.argparser import get_test_args
from utils.dataset import RGBT_Dataset
from utils.utils import save_some_examples, load_checkpoint, calculate_psnr
from models.generator import UNet as Generator


def test():
    args = get_test_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)


    os.makedirs(args.save_dir, exist_ok=True)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load generator
    generator = Generator(
        in_channels=3,
        out_channels=1,
        base_filters=args.gen_filters,
        upsampling_method=args.upsampling_method
    ).to(device)

    checkpoint_path = os.path.join(args.save_dir, "checkpoints", "gen_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(checkpoint_path, generator)

    generator.eval()

    # Transformations (match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((300, 480))
    ])

    test_dataset = RGBT_Dataset(
        data_dir=args.data,
        sr=args.sr,
        thermal_type=args.thermal_type,
        mode='test',
        transform=transform, random_state=args.random_state, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    os.makedirs(os.path.join(args.save_dir, "test_results"), exist_ok=True)

    with torch.no_grad():
        for batch_idx, (input_img, target_img) in enumerate(tqdm(test_loader, desc="Testing")):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            gen_output = generator(input_img)

            batch_size = input_img.size(0)

            psnr_val = calculate_psnr(gen_output, target_img)
            ssim_val = structural_similarity_index_measure(gen_output, target_img, data_range=1.0)

            total_psnr += psnr_val.item() * batch_size
            total_ssim += ssim_val.item() * batch_size
            total_samples += batch_size

            # Optional: save example outputs
            if batch_idx == 0:
                save_some_examples(
                    generator, test_loader, 0, 
                    folder=os.path.join(args.save_dir, "test_results"),
                    device=device,
                    denorm=False
                )

    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    print(f"\nTest Results — Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")

    # write the test results to a file
    with open(os.path.join(args.save_dir, "test_results", "test_results.txt"), "w") as f:
        f.write(f"Avg PSNR: {avg_psnr:.4f}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Random State: {args.random_state}\n")
        f.write(f"Data Directory: {args.data}\n")
        f.write(f"Thermal Type: {args.thermal_type}\n")
        f.write(f"SR: {args.sr}\n")
        f.write(f"Train Ratio: {args.train_ratio}\n")
        f.write(f"Validation Ratio: {args.val_ratio}\n")
        f.write(f"Generator Filters: {args.gen_filters}\n")
        f.write(f"Upsampling Method: {args.upsampling_method}\n")
        f.write(f"Save Directory: {args.save_dir}\n")
        f.write(f"Checkpoint Path: {checkpoint_path}\n")


if __name__ == "__main__":
    test()
