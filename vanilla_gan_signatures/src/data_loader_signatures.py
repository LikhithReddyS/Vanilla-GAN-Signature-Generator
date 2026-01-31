"""
Data Loader for Signature Dataset
=================================
Handles loading of signature images from directory.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SignatureDataset(Dataset):
    """
    Custom Dataset for loading Signature images.
    Assumes images are already preprocessed/sized or handles resizing if not.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.image_paths = list(self.root_dir.glob('**/*.png')) + \
                           list(self.root_dir.glob('**/*.jpg')) + \
                           list(self.root_dir.glob('**/*.jpeg'))
        
        self.transform = transform
        
        # Default transform if none provided: Grayscale, Resize, CenterCrop, Tensor, Normalize [-1, 1]
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # Normalize [0,1] to [-1,1]
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L') # Ensure Grayscale
            
            if self.transform:
                image = self.transform(image)
                
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle error appropriately. 
            # For simplicity in training loops, we might want to just skip or return zeros.
            # Returning zeros to avoid breaking the batch
            return torch.zeros((1, 128, 128))

def get_data_loader(data_dir, batch_size=32, num_workers=2, shuffle=True, pin_memory=False):
    """
    Helper function to get DataLoader.
    
    Args:
        data_dir (str): Path to data directory.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        shuffle (bool): Whether to shuffle data.
        pin_memory (bool): Use pinned memory for faster GPU transfer.
        
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = SignatureDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader
