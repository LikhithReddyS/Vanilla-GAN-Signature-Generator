# Vanilla GAN for Synthetic Handwritten Signature Generation

## Overview
This project implements an unconditional **Vanilla GAN** to generate realistic 64x64 grayscale handwritten signatures. The goal is to investigate whether synthetic signatures can be used to augment datasets and improve the robustness of signature verification systems.

The project includes:
- **Data Pipeline**: Loading and preprocessing (resize, grayscale, normalize).
- **GAN Architecture**: A custom Generator and Discriminator optimized for 64x64 images.
- **Training Engine**: Robust training loop with Logger and Visualizer.
- **Evaluation**: Tools to compute Accuracy, FAR, FRR, and EER.
- **Verification Experiment**: A baseline verifier (CNN) to test Real vs. Augmented datasets.
- **UI**: A Gradio app for easy interaction.

## Directory Structure
```
vanilla_gan_signatures/
 ├── src/                 # Source code
 │   ├── models/          # GAN and Verifier models
 │   ├── utils/           # Helper scripts (logger, metrics)
 │   ├── train_*.py       # Training scripts
 │   └── app_*.py         # Gradio App
 ├── data/                # Place raw images here
 ├── checkpoints/         # Saved models
 ├── samples/             # Generated samples during training
 └── logs/                # Training logs
```

## Architecture
### Generator
Linear(100 -> 8192) -> Reshape(512x4x4) -> ConvTranspose2d Blocks -> Tanh -> Image(64x64)
```
[Noise z] -> [Dense] -> [Reshape] -> [DeConv] -> [DeConv] -> [DeConv] -> [DeConv] -> [Output]
```

### Discriminator
Image(64x64) -> Conv2d Blocks -> Flatten -> Dense -> Sigmoid -> Probability
```
[Image] -> [Conv] -> [Conv] -> [Conv] -> [Conv] -> [Flatten] -> [Dense] -> [Real/Fake]
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV, Numpy, Matplotlib, Scikit-learn, Gradio

### 1. Preprocessing
Prepare your signature dataset (e.g., CEDAR, MCYT, or custom).
```bash
python -m src.preprocess_signatures --input_dir path/to/raw --output_dir data/processed
```

### 2. Train GAN
```bash
python -m src.train_vanilla_gan_signatures --epochs 100 --batch_size 64 --data_dir data/processed
```

### 3. Generate Signatures
```bash
python -m src.generate_signatures --checkpoint_path checkpoints/generator_final.pth --num_images 100
```

### 4. Verification Experiment
Train a verifier on Real data only:
```bash
python -m src.signature_verifier_train --real_data_dir data/processed --checkpoint_dir checkpoints
```
Train with Augmentation:
```bash
python -m src.signature_verifier_train --real_data_dir data/processed --synthetic_data_dir data/synthetic --use_augmentation
```

### 5. Evaluate
```bash
python -m src.signature_verifier_eval --model_path checkpoints/verifier_baseline.pth --test_data_dir data/test_processed
```

### 6. Run Demo App
```bash
streamlit run src/app_vanilla_gan_signatures.py
```

## Results & Evaluation
The system computes:
- **FAR (False Acceptance Rate)**: Likelihood of accepting a forgery.
- **FRR (False Rejection Rate)**: Likelihood of rejecting a genuine signature.
- **EER (Equal Error Rate)**: The point where FAR = FRR.

Lower EER indicates better performance.

## Future Work
- Implement Conditional GAN (cGAN) to generate signatures for specific user IDs.
- Use larger image sizes (128x128).
- Explore CycleGAN for offline-to-online signature conversion.
