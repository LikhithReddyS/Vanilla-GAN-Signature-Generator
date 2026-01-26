"""
Logger
======
Logging utility for training progress.
"""

import os
import csv
import time

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.csv')
        
        # Initialize CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Batch', 'D_Loss', 'G_Loss', 'Time'])

    def log(self, epoch, batch, d_loss, g_loss):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, f"{d_loss:.4f}", f"{g_loss:.4f}", time.time()])
        
        # Also print to console
        # print(f"[Epoch {epoch}] [Batch {batch}] [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
