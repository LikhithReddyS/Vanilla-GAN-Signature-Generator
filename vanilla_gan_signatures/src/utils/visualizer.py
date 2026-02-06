"""
Visualizer
==========
Visualization helper functions.
"""

import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def save_sample_images(generator, z, device, save_path, show_in_colab=False, epoch=None):
    """
    Generate and save a grid of images.
    If show_in_colab=True, also display inline in Colab/Jupyter.
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(z)
    generator.train()
    
    # Denormalize: [-1, 1] -> [0, 1]
    vutils.save_image(fake_images, save_path, normalize=True, padding=2)
    
    # Display in Colab if requested
    if show_in_colab:
        display_sample_images(fake_images, epoch)

def display_sample_images(images, epoch=None):
    """
    Display a grid of images inline (for Colab/Jupyter).
    Uses IPython.display for immediate real-time display during training.
    """
    import sys
    
    # Make grid
    grid = vutils.make_grid(images, normalize=True, padding=2)
    # Convert to numpy for matplotlib
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(12, 12))
    title = f"Generated Signatures - Epoch {epoch}" if epoch is not None else "Generated Signatures"
    plt.title(title, fontsize=14)
    plt.imshow(grid_np, cmap='gray')
    plt.axis('off')
    
    # Use IPython display for immediate output in Colab
    try:
        from IPython.display import display, clear_output
        plt.show()
        sys.stdout.flush()  # Force output to appear immediately
    except ImportError:
        plt.show()
    
    plt.close()  # Close figure to free memory

def save_loss_plot(d_losses, g_losses, save_path):
    """
    Save loss plot.
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses, label="G")
    plt.plot(g_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

