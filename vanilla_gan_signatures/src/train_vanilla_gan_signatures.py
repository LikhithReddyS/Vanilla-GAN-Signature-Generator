"""
Train Vanilla GAN
=================
Main training loop for the GAN.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.vanilla_gan_model import VanillaGAN
from src.data_loader_signatures import get_data_loader
from src.utils.logger import Logger
from src.utils.visualizer import save_sample_images, save_loss_plot

def train(args):
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loader
    dataloader = get_data_loader(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Model
    gan = VanillaGAN(z_dim=args.z_dim, im_channels=1, device=device)
    generator, discriminator = gan.get_models()
    
    # Optimizers
    # Beta1 = 0.5 is important for stable GAN training
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.z_dim, device=device)
    
    # Logging
    logger = Logger(args.log_dir)
    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    
    for epoch in range(args.epochs):
        for i, real_imgs in enumerate(dataloader):
            
            # Configure input
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)
            
            # Label Smoothing
            # Real labels are 0.9 instead of 1.0 to help discriminator generalization
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((current_batch_size, 1), 0.0, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Train with Real
            output_real = discriminator(real_imgs)
            d_loss_real = criterion(output_real, real_labels)
            
            # Train with Fake
            z = torch.randn(current_batch_size, args.z_dim, device=device)
            fake_imgs = generator(z)
            output_fake = discriminator(fake_imgs.detach()) # Detach G gradients
            d_loss_fake = criterion(output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # We want G to fool D, so we use real_labels (1.0 ideally, or smoothed)
            # Usually for G loss we target 1.0 strict
            target_labels = torch.full((current_batch_size, 1), 1.0, device=device)
            
            output_fake_for_g = discriminator(fake_imgs) # Keep gradients
            g_loss = criterion(output_fake_for_g, target_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Save Losses config
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            # Logging
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                logger.log(epoch, i, d_loss.item(), g_loss.item())
        
        # End of Epoch
        # Save samples
        save_path = os.path.join(args.sample_dir, f"epoch_{epoch}.png")
        save_sample_images(generator, fixed_noise, device, save_path)
        
        # Save Checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_epoch_{epoch}.pth"))
            
    # Save Final Model
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "discriminator_final.pth"))
    
    # Save Loss Plot
    save_loss_plot(D_losses, G_losses, os.path.join(args.figures_dir, "training_loss.png"))
    print("Training Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="path to processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="path to save checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples", help="path to save sample images")
    parser.add_argument("--log_dir", type=str, default="logs", help="path to save logs")
    parser.add_argument("--figures_dir", type=str, default="figures", help="path to save figures")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    
    args = parser.parse_args()
    train(args)
