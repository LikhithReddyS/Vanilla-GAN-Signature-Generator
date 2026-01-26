"""
Vanilla GAN Generator
=====================
Generator architecture for 64x64 signature generation.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, im_channels=1):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        
        # Initial dense layer to project z to a feature map volume
        # We start with 4x4 feature maps. 
        # Target: 64x64. Upsampling steps: 4->8->16->32->64 (4 Upsamples)
        self.init_size = 4
        self.feature_maps = 512 # Start with many filters
        
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, self.feature_maps * self.init_size ** 2),
            nn.BatchNorm1d(self.feature_maps * self.init_size ** 2), # Optional for linear, but helpful
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(self.feature_maps, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, im_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output in [-1, 1]
        )
    
    def forward(self, z):
        # z: (batch, z_dim)
        out = self.l1(z)
        out = out.view(out.shape[0], self.feature_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

if __name__ == '__main__':
    # Simple test
    z = torch.randn(10, 100)
    gen = Generator()
    out = gen(z)
    print(f"Generator output shape: {out.shape}")
