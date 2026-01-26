"""
Generate Signatures
===================
Script to generate synthetic signatures using trained Generator.
"""

import argparse
import torch
import torchvision.utils as vutils
import os
from src.generator_vanilla_gan import Generator

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Generator
    generator = Generator(z_dim=args.z_dim).to(device)
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}")
        return
        
    generator.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    generator.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} images...")
    
    with torch.no_grad():
        for i in range(args.num_images):
            z = torch.randn(1, args.z_dim, device=device)
            fake_img = generator(z)
            
            # Save individual images for dataset augmentation
            save_path = os.path.join(args.output_dir, f"synthetic_sig_{i}.png")
            vutils.save_image(fake_img, save_path, normalize=True)
            
    print(f"Done. Saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--z_dim", type=int, default=100, help="Latent dimension")
    
    args = parser.parse_args()
    generate(args)
