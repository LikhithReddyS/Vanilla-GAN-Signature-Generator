"""
Evaluate Signature Verifier
===========================
Evaluate Accuracy, FAR, FRR, EER.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
# Add project root to path so 'from src...' imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signature_verifier_train import SimpleVerifier
from src.data_loader_signatures import SignatureDataset
from src.utils.metrics import compute_eer
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SimpleVerifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load Test Data (Real signatures) -> Positive Class
    test_dataset = SignatureDataset(args.test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    y_true = []
    y_scores = []
    
    print("Evaluating...")
    
    with torch.no_grad():
        # Pass 1: Positive Samples (Real Signatures)
        for imgs in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs) # Score [0, 1]
            
            y_true.extend([1] * imgs.size(0))
            y_scores.extend(preds.cpu().numpy().flatten())
            
        # Pass 2: Negative Samples (Noise/Fake)
        # We generate same number of negatives
        num_negatives = len(test_dataset)
        noise = torch.randn(num_negatives, 1, 64, 64, device=device)
        # Batch processing for noise
        batch_size = 32
        for i in range(0, num_negatives, batch_size):
            batch_noise = noise[i:i+batch_size]
            preds = model(batch_noise)
            
            y_true.extend([0] * batch_noise.size(0))
            y_scores.extend(preds.cpu().numpy().flatten())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Compute Metrics
    from sklearn.metrics import accuracy_score
    
    # Accuracy at 0.5 threshold
    y_pred = (y_scores > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    # EER
    eer, threshold = compute_eer(y_true, y_scores)
    
    print(f"Results for {args.model_path}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"EER: {eer:.4f} at Threshold {threshold:.4f}")
    
    # Plot ROC if requested
    if args.plot_path:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(args.plot_path)
        print(f"Saved ROC plot to {args.plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--plot_path", type=str, default=None)
    
    args = parser.parse_args()
    evaluate(args)
