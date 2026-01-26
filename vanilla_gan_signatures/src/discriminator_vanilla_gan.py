"""
Vanilla GAN Discriminator
=========================
Discriminator architecture for 64x64 signature generation.
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, im_channels=1):
        super(Discriminator, self).__init__()
        
        # Input 64x64
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(im_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (No BatchNorm in the first layer usually)
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Flatten and Dense
        # 4*4*512 = 8192
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        # img: (batch, channels, 64, 64)
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1) # Flatten
        validity = self.adv_layer(out)
        return validity

if __name__ == '__main__':
    # Simple test
    img = torch.randn(10, 1, 64, 64)
    disc = Discriminator()
    out = disc(img)
    print(f"Discriminator output shape: {out.shape}")
