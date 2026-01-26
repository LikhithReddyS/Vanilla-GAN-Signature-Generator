"""
Train Signature Verifier
========================
Train a baseline classifier/verifier on real (and optionally synthetic) data.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from src.data_loader_signatures import SignatureDataset

class SimpleVerifier(nn.Module):
    """
    Simple Binary Classifier for Signatures: Genuine (1) vs Forgery (0).
    Note: Ideally for signature verification we use Siamese networks to compare pairs.
    However, for this task, the prompt implies "Train verifier on Real vs Real + Synthetic".
    If we treat Synthetic as "Augmentation" for the "Real" class (Genuine), we can check if it improves robustness.
    Or if we treat Synthetic as "Forged", we try to detect them.
    
    The prompt says: "Augment a signature verification system".
    Usually this means we train a classifier to distinguish User A from User B, or Genuine vs Forged.
    Since we don't have "Forged" labels in an unconditional GAN context (unless we consider random noise or other signatures as forgeries),
    a common setup is One-Class Classification or Standard Classification if we have multi-class.
    
    Given the prompt "Unconditional GAN to learn handwritten signature distributions",
    it likely generates "Generic Signatures".
    If we verify "Realness", we can train a classifier Real vs Synthetic? 
    OR, if we assume we have a dataset of signatures and we want to classify them.
    
    Let's assume a simplified Verification Task:
    We treat the task as a binary classification (Real vs Random Noise/Other) or Multi-class (User ID).
    However, with Unconditional GAN, we just get "A signature".
    
    INTERPRETATION:
    The most logical "Augmentation" experiment with Unconditional GANs is:
    Train a classifier to distinguish "Signature" vs "Non-Signature" (Robustness)
    OR
    If we had conditional labels (User 1, User 2), we could augment User 1's data. But we don't.
    
    Let's implement a binary classifier that tries to distinguish [Real Signature] vs [Synthetic/Fake Signature].
    Wait, usually we want to IMPROVE accuracy.
    
    Let's assume the goal is "Signature vs Non-Signature/Noise" or just training a robust feature extractor.
    
    BETTER INTERPRETATION based on "Verification System":
    Typically, Verification = Is this image Signature of User X?
    Without user labels, we arguably cannot build a standard verification system.
    
    Compromise for this "Research Project":
    We will simulate a "Real vs Fake" detector (Discriminator-like) as the "Verification System" to detect forgeries.
    The "Augmentation" part: We mix Real + Synthetic to train a better detector? No, that's circular.
    
    Let's stick to the prompt: "Use synthetic signatures to augment a signature verification system".
    This implies we have a labeled dataset (e.g., CEDAR, MCYT) with Forgeries.
    Since we don't have an external dataset, we will assume we are training a model to detect "Forged" (Random/Bad) from "Genuine" (Real).
    We can treat Synthetic signatures as "Potential Forgeries" to make the system robust?
    OR we treat Synthetic signatures as "Genuine" samples to increase training size (Augmentation).
    
    Let's go with **Augmentation**:
    We assume all Real images are class 1 (Genuine).
    We assume we have some class 0 (Forged) - maybe we generate random noise or invert images? 
    Or maybe we just train an Autoencoder/One-Class SVM?
    
    Let's Implement a simple Convolutional Classifier.
    We will facilitate training on:
    1. Real Data Only
    2. Real + Synthetic Data (all labeled as Real)
    
    And we test on a held-out set of Real Data (to see if accuracy improves? No, that's trivial).
    
    Actually, **Outlier Detection** (Anomaly Detection) is best.
    Train on Real (One class). Test on Real vs Synthetic (to see if Synthetic is accepted).
    BUT, the goal is to evaluate impact on Accuracy/FAR/FRR.
    
    Let's assume we split Real Data into Train/Test.
    We train a classifier to distinguish "Real Train" from "Noise/Random".
    Then we augment "Real Train" with "Synthetic".
    We test on "Real Test" (Should be classified as Real) and "Noise/Random" (Classified as Fake).
    
    This is a bit synthetic but fits the Unconditional GAN constraint.
    """
    def __init__(self):
        super(SimpleVerifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_verifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Real Data
    real_dataset = SignatureDataset(args.real_data_dir)
    
    # 2. Load Synthetic Data (if augmentation enabled)
    if args.use_augmentation and args.synthetic_data_dir:
        synth_dataset = SignatureDataset(args.synthetic_data_dir)
        # Combine
        combined_dataset = ConcatDataset([real_dataset, synth_dataset])
        print(f"Training with Augmentation. Real: {len(real_dataset)}, Synth: {len(synth_dataset)}")
        dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"Training on Real Data Only. Size: {len(real_dataset)}")
        dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True)
        
    # Model
    model = SimpleVerifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("Starting Verifier Training...")
    model.train()
    
    # Since we only have "Real" class (1.0), this is trivial unless we add "Negative" samples.
    # To make this a valid classification task (FAR/FRR), we need Negatives.
    # Let's generate "Negative" samples on the fly: Random Noise or Inverted images.
    
    for epoch in range(args.epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # Positive samples (Real/Synthetic Signatures) -> Label 1
            targets_pos = torch.ones(batch_size, 1, device=device)
            preds_pos = model(imgs)
            loss_pos = criterion(preds_pos, targets_pos)
            
            # Negative samples (Random Noise) -> Label 0
            noise = torch.randn_like(imgs)
            targets_neg = torch.zeros(batch_size, 1, device=device)
            preds_neg = model(noise)
            loss_neg = criterion(preds_neg, targets_neg)
            
            loss = loss_pos + loss_neg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss / len(dataloader):.4f}")
        
    # Save
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    name = "verifier_augmented.pth" if args.use_augmentation else "verifier_baseline.pth"
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, name))
    print(f"Saved model to {os.path.join(args.checkpoint_dir, name)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_dir", type=str, required=True)
    parser.add_argument("--synthetic_data_dir", type=str, default=None)
    parser.add_argument("--use_augmentation", action='store_true')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    train_verifier(args)
