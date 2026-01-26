"""
Streamlit App
=============
UI for exploring the GAN and generating signatures.
"""

import streamlit as st
import torch
import os
import sys
import torchvision.transforms as transforms
from PIL import Image

# --------------------------------------------------
# PATH SETUP (IMPORTANT FOR STREAMLIT CLOUD)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # vanilla_gan_signatures/src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # vanilla_gan_signatures

sys.path.append(PROJECT_ROOT)

from src.generator_vanilla_gan import Generator

# --------------------------------------------------
# STREAMLIT PAGE CONFIG (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="Vanilla GAN Signature Generator",
    layout="wide"
)

# --------------------------------------------------
# MODEL LOADING FUNCTION
# --------------------------------------------------
def load_generator(checkpoint_path, z_dim, device):
    if not os.path.exists(checkpoint_path):
        st.error("❌ Model checkpoint not found in repository.")
        st.stop()

    try:
        generator = Generator(z_dim=z_dim)
        generator.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        generator.to(device)
        generator.eval()
        return generator

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():
    st.title("✍️ Vanilla GAN Signature Generator")
    st.markdown(
        "Generate synthetic **64×64 grayscale handwritten signatures** "
        "using a trained **Vanilla GAN**."
    )

    # --------------------------------------------------
    # CHECKPOINT PATH (CLOUD SAFE)
    # --------------------------------------------------
    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "generator_final.pth"
    )

    # --------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------
    st.sidebar.header("Configuration")
    st.sidebar.info("Using pre-trained GAN checkpoint from repository")

    num_signatures = st.sidebar.slider(
        "Number of Signatures",
        min_value=1,
        max_value=100,
        value=8
    )

    generate_btn = st.sidebar.button("Generate Signatures")

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------
    if generate_btn:
        generator = load_generator(checkpoint_path, z_dim, device)

        with st.spinner("Generating signatures..."):
            with torch.no_grad():
                z = torch.randn(num_signatures, z_dim, device=device)
                fake_imgs = generator(z)

                images = []
                for img_tensor in fake_imgs:
                    # Denormalize from [-1, 1] → [0, 1]
                    img = (img_tensor.cpu() + 1) / 2
                    img = img.clamp(0, 1)
                    img = transforms.ToPILImage()(img)
                    images.append(img)

        st.success(f"Generated {num_signatures} signatures!")

        # --------------------------------------------------
        # DISPLAY IMAGES
        # --------------------------------------------------
        cols = st.columns(4)
        for i, img in enumerate(images):
            with cols[i % 4]:
                st.image(
                    img,
                    use_container_width=True,
                    caption=f"Sig {i + 1}"
                )

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
