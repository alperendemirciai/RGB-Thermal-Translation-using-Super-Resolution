import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import numpy as np

from utils.argparser import get_train_val_args
from utils.dataset import RGBT_Dataset

from utils.utils import *

from models.generator import UNet as Generator
from models.discriminator import Discriminator

from torchmetrics.functional import structural_similarity_index_measure


def train_validate():
    args = get_train_val_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    os.makedirs(args.save_dir, exist_ok=True)

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Initialize models
    generator = Generator(
        in_channels=3,
        out_channels=1,
        base_filters=args.gen_filters,
        upsampling_method=args.upsampling_method
    ).to(device)

    discriminator = Discriminator(
        in_channels=3,
        target_channels=1,
        base_filters=args.disc_filters
    ).to(device)
    
    # Initialize optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs)

    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((300, 480)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    train_dataset = RGBT_Dataset(
        data_dir=args.data,
        sr=args.sr,
        thermal_type=args.thermal_type,
        mode='train',
        transform=transform,
        random_state=args.random_state,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    val_dataset = RGBT_Dataset(
        data_dir=args.data,
        sr=args.sr,
        thermal_type=args.thermal_type,
        mode='val',
        transform=transform,
        random_state=args.random_state,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    # History for tracking metrics
    history = {
        'train_gen_loss': [],
        'train_disc_loss': [],
        'train_pixel_loss': [],
        'val_gen_loss': [],
        'val_disc_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_pixel_loss': []
    }
    
    # Best validation metrics
    best_val_psnr = 0
    best_epoch = 0

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        # Training phase
        generator.train()
        discriminator.train()
        
        # Initialize epoch metrics
        train_gen_loss = 0
        train_disc_loss = 0
        train_pixel_loss = 0
        
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (input_img, target_img) in enumerate(loop):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
            # Determine batch size (last batch might be smaller)
            batch_size = input_img.size(0)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            gen_output = generator(input_img)
            
            # Calculate GAN loss (fool discriminator)
            pred_fake = discriminator(input_img, gen_output)
            valid = torch.ones(batch_size, 1, pred_fake.size(2), pred_fake.size(3)).to(device)
            fake = torch.zeros(batch_size, 1, pred_fake.size(2), pred_fake.size(3)).to(device)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Calculate pixel-wise loss
            loss_pixel = criterion_pixel(gen_output, target_img)
            
            # Total loss
            loss_G = loss_GAN + 100 * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(input_img, target_img)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(input_img, gen_output.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            # Update progress bar
            loop.set_postfix(
                D_real=loss_real.item(),
                D_fake=loss_fake.item(),
                G_adv=loss_GAN.item(),
                G_pixel=loss_pixel.item(),
            )
            
            # Accumulate batch loss for epoch metrics
            train_gen_loss += loss_G.item() * batch_size
            train_disc_loss += loss_D.item() * batch_size
            train_pixel_loss += loss_pixel.item() * batch_size
        

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()


        # Calculate average epoch losses
        train_gen_loss /= len(train_dataset)
        train_disc_loss /= len(train_dataset)
        train_pixel_loss /= len(train_dataset)
        
        # Store training metrics
        history['train_gen_loss'].append(train_gen_loss)
        history['train_disc_loss'].append(train_disc_loss)
        history['train_pixel_loss'].append(train_pixel_loss)

        # ---------------------
        #  Validation Phase
        # ---------------------
        generator.eval()
        discriminator.eval()
        
        val_gen_loss = 0
        val_psnr_avg = 0
        val_ssim_avg = 0
        val_disc_loss = 0
        val_pixel_loss = 0
        loop = tqdm(val_loader, leave=True, desc=f"Validation {epoch+1}/{args.epochs}")
        
        with torch.no_grad():
            for batch_idx, (input_img, target_img) in enumerate(loop):
                input_img = input_img.to(device)
                target_img = target_img.to(device)
                
                # Determine batch size
                batch_size = input_img.size(0)
                
                # Generate fake images
                gen_output = generator(input_img)
                
                # Calculate losses
                pred_fake = discriminator(input_img, gen_output)
                valid = torch.ones(batch_size, 1, pred_fake.size(2), pred_fake.size(3)).to(device)
                fake = torch.zeros(batch_size, 1, pred_fake.size(2), pred_fake.size(3)).to(device)
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixel(gen_output, target_img)
                loss_G = loss_GAN + 100 * loss_pixel
                
                # Calculate PSNR
                psnr_val = calculate_psnr(gen_output, target_img)
                ssim_val = structural_similarity_index_measure(gen_output, target_img, data_range=1.0)

                print(f"Validation PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                # Update progress bar
                loop.set_postfix(
                    G_adv=loss_GAN.item(),
                    G_pixel=loss_pixel.item(),
                    PSNR=psnr_val.item(),
                    SSIM=ssim_val.item()
                )
                
                # Accumulate batch metrics
                val_gen_loss += loss_G.item() * batch_size
                val_psnr_avg += psnr_val.item() * batch_size
                val_ssim_avg += ssim_val.item() * batch_size
                val_disc_loss += loss_D.item() * batch_size
                val_pixel_loss += loss_pixel.item() * batch_size
        
        # Calculate average validation metrics
        val_gen_loss /= len(val_dataset)
        val_psnr_avg /= len(val_dataset)
        val_ssim_avg /= len(val_dataset)
        val_disc_loss /= len(val_dataset)
        val_pixel_loss /= len(val_dataset)
        
        # Store validation metrics
        history['val_gen_loss'].append(val_gen_loss)
        history['val_psnr'].append(val_psnr_avg)
        history['val_ssim'].append(val_ssim_avg)
        history['val_disc_loss'].append(val_disc_loss)
        history['val_pixel_loss'].append(val_pixel_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"Train Gen Loss: {train_gen_loss:.4f}, Train Disc Loss: {train_disc_loss:.4f}")
        print(f"Validation Gen Loss: {val_gen_loss:.4f}, Validation PSNR: {val_psnr_avg:.4f} dB")
        
        # Check if this is the best model so far
        if val_psnr_avg > best_val_psnr:
            best_val_psnr = val_psnr_avg
            best_epoch = epoch + 1

            os.makedirs(os.path.join(args.save_dir,"checkpoints"), exist_ok=True)
            
            # Save best model
            best_checkpoint_gen = os.path.join(args.save_dir,"checkpoints", "gen_best.pth")
            best_checkpoint_disc = os.path.join(args.save_dir,"checkpoints", "disc_best.pth")
            
            save_checkpoint(generator, optimizer_G, epoch + 1, best_checkpoint_gen)
            save_checkpoint(discriminator, optimizer_D, epoch + 1, best_checkpoint_disc)
            print(f"Saved best model at epoch {epoch+1} with PSNR: {best_val_psnr:.4f} dB")
        
        # Save model checkpoints periodically
        if (epoch + 1) % args.save_freq == 0:
            """
            checkpoint_gen = os.path.join("checkpoints", f"gen_epoch_{epoch+1}.pth")
            checkpoint_disc = os.path.join("checkpoints", f"disc_epoch_{epoch+1}.pth")
            
            save_checkpoint(generator, optimizer_G, epoch + 1, checkpoint_gen)
            save_checkpoint(discriminator, optimizer_D, epoch + 1, checkpoint_disc)
            """
            # Save example outputs
            os.makedirs(os.path.join(args.save_dir,"results"), exist_ok=True)

            save_some_examples(
                generator, 
                val_loader,  # Use validation set for examples
                epoch + 1, 
                folder=os.path.join(args.save_dir,"results"), 
                device=device, 
                denorm=False
            )
            """
            # Save latest model
            latest_checkpoint_gen = os.path.join("checkpoints", "gen_latest.pth")
            latest_checkpoint_disc = os.path.join("checkpoints", "disc_latest.pth")
            
            save_checkpoint(generator, optimizer_G, epoch + 1, latest_checkpoint_gen)
            save_checkpoint(discriminator, optimizer_D, epoch + 1, latest_checkpoint_disc)
            """
        
    
    print(f"Training completed. Best model at epoch {best_epoch} with PSNR: {best_val_psnr:.4f} dB")

    os.makedirs(os.path.join(args.save_dir,"results"), exist_ok=True)

    plot_metrics(
        history['train_gen_loss'],
        history['val_gen_loss'],
        'Generator Loss',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metrics(
        history['train_pixel_loss'],
        history['val_pixel_loss'],
        'Pixel Loss',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metrics(
        history['train_disc_loss'],
        history['val_disc_loss'],
        'Discriminator Loss',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metric(
        history['val_psnr'],
        'Validation PSNR',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metric(
        history['val_ssim'],
        'Validation SSIM',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    
    # Save training history to a file
    history_file = os.path.join(args.save_dir, "training_history.txt")
    with open(history_file, 'w') as f:
        for key, values in history.items():
            f.write(f"{key}: {values}\n")
    print(f"Training history saved to {history_file}")

    # save training params to a file
    params_file = os.path.join(args.save_dir, "training_params.txt")
    with open(params_file, 'w') as f:
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Generator Filters: {args.gen_filters}\n")
        f.write(f"Discriminator Filters: {args.disc_filters}\n")
        f.write(f"Upsampling Method: {args.upsampling_method}\n")
        f.write(f"Random State: {args.random_state}\n")
        f.write(f"Train Ratio: {args.train_ratio}\n")
        f.write(f"Validation Ratio: {args.val_ratio}\n")
        f.write(f"Data Directory: {args.data}\n")
        f.write(f"Thermal Type: {args.thermal_type}\n")
        f.write(f"SR: {args.sr}\n")
        f.write(f"Save Directory: {args.save_dir}\n")
        f.write(f"Learning Rate {args.lr}\n")
        print(f"Training parameters saved to {params_file}")


if __name__ == '__main__':
    train_validate()
