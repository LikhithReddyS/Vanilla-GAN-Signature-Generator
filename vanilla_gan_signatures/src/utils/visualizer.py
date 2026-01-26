"""
Visualizer
==========
Visualization helper functions.
"""

import os
import torch
import torchvision.utils as vutils
import numpy as np

def save_sample_images(generator, z, device, save_path):
    """
    Generate and save a grid of images.
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(z)
    generator.train()
    
    # Denormalize: [-1, 1] -> [0, 1]
    vutils.save_image(fake_images, save_path, normalize=True, padding=2)

def save_loss_plot(d_losses, g_losses, save_path):
    """
    Save loss plot.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses, label="G")
    plt.plot(g_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
