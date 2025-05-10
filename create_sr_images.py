import torch
from PIL import Image
from torchvision import transforms
from torchsr.models import edsr
import os
from glob import glob

#TODO: x2, add to the dataset.py
#NOTE: obsolete, but keep for reference

def apply_super_resolution_to_all_images():
    # Set up device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Directory containing input images
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, "labeled_rgbt_pairs", "color")
    output_dir = input_dir  # Save in same folder

    # List all .jpg files in the input directory
    image_paths = glob(os.path.join(input_dir, "*.jpg"))

    # Define transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Load EDSR model (2× SR)
    model = edsr(scale=2, pretrained=True).to(device)
    model.eval()

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(img_tensor).squeeze(0).clamp(0.0, 1.0)

        original_size = img_tensor.shape[-2:]  # (H, W)
        sr_resized_tensor = torch.nn.functional.interpolate(
            sr_tensor.unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Convert to PIL
        sr_img_resized = to_pil(sr_resized_tensor.cpu())

        # Construct output file name
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_sr.jpg")
        sr_img_resized.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    apply_super_resolution_to_all_images()
