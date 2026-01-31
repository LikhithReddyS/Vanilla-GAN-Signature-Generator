"""
Vanilla GAN Model Wrapper
=========================
Wrapper  class to handle Generator and Discriminator interaction.
"""

import torch
import torch.nn as nn
from src.generator_vanilla_gan import Generator
from src.discriminator_vanilla_gan import Discriminator

class VanillaGAN:
    def __init__(self, z_dim=100, im_channels=1, device='cpu'):
        self.z_dim = z_dim
        self.device = device
        
        self.generator = Generator(z_dim, im_channels).to(device)
        self.discriminator = Discriminator(im_channels).to(device)
        
        self.reinit_weights()
        
    def reinit_weights(self):
        """
        Initialize weights: Normal distribution with mean=0, std=0.02
        As suggested in DCGAN paper.
        """
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.z_dim, device=self.device)

    def get_models(self):
        return self.generator, self.discriminator
