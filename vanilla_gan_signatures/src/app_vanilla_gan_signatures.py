"""
Streamlit App
=============
UI for exploring the GAN and generating signatures.
"""

import streamlit as st
import torch
import torchvision.utils as vutils
import numpy as np
import os
import sys
# Add project root to path so 'from src...' imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator_vanilla_gan import Generator
import torchvision.transforms as transforms
from PIL import Image

# Page Config
st.set_page_config(page_title="Vanilla GAN Signature Generator", layout="wide")

def load_generator(checkpoint_path, z_dim, device):
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        generator = Generator(z_dim=z_dim).to(device)
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
        generator.eval()
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Vanilla GAN Signature Generator")
    st.markdown("Generate synthetic 64x64 grayscale handwritten signatures using a trained Vanilla GAN.")
    
    # Determine default checkpoint path relative to this script
    # Script is in vanilla_gan_signatures/src/
    # Checkpoints are in vanilla_gan_signatures/checkpoints/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ckpt_path = os.path.join(base_dir, "checkpoints", "generator_final.pth")
    
    # Sidebar Controls
    st.sidebar.header("Configuration")
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value=default_ckpt_path)
    num_signatures = st.sidebar.slider("Number of Signatures", min_value=1, max_value=100, value=8)
    
    generate_btn = st.sidebar.button("Generate Signatures")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100
    
    if generate_btn:

        generator = load_generator(checkpoint_path, z_dim, device)
        
        if generator:
            with st.spinner("Generating signatures..."):
                with torch.no_grad():
                    z = torch.randn(int(num_signatures), z_dim, device=device)
                    fake_imgs = generator(z)
                    
                    # Convert to PIL images
                    images = []
                    for img_tensor in fake_imgs:
                        # Denormalize: [-1, 1] -> [0, 1] -> [0, 255]
                        img = (img_tensor.cpu() + 1) / 2
                        img = img.clamp(0, 1)
                        img = transforms.ToPILImage()(img)
                        images.append(img)
            
            st.success(f"Generated {num_signatures} signatures!")
            
            # Display in columns
            cols = st.columns(4)
            for i, img in enumerate(images):
                with cols[i % 4]:
                    st.image(img, use_column_width=True, caption=f"Sig {i+1}")

if __name__ == "__main__":
    main()
