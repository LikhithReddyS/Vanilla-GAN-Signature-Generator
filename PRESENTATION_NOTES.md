# Project Presentation: Vanilla GAN for Synthetic Handwritten Signature Generation

## 1. Project Overview
*   **Title**: Vanilla GAN for Synthetic Handwritten Signature Generation
*   **Goal**: Generate realistic 64x64 grayscale handwritten signatures using Generative Adversarial Networks (GANs).
*   **Purpose**: To augment limited signature datasets and improve the training of signature verification systems.

## 2. Problem Statement
*   **Data Scarcity**: Gathering large datasets of genuine signatures is difficult and privacy-sensitive.
*   **Model Robustness**: Verification systems need diverse examples to generalize well.
*   **Solution**: Use GANs to synthesize new, unique signature samples that mimic the style of real handwriting.

## 3. Methodology & Architecture
The project uses an **Unconditional Vanilla GAN** architecture.

### A. Generator (The Artist)
*   **Input**: Random noise vector (latent space, size 100).
*   **Process**: 
    1.  Fully Connected Layer (Dense) -> Reshape.
    2.  Transposed Convolutional Layers (Upsampling).
    3.  Activation Functions (ReLU, Tanh at output).
*   **Output**: A 64x64 grayscale image (pretending to be a signature).

### B. Discriminator (The Critic)
*   **Input**: An image (64x64), either Real (from dataset) or Fake (from Generator).
*   **Process**:
    1.  Convolutional Layers (Downsampling).
    2.  Flatten -> Fully Connected Layer.
    3.  Activation Function (Sigmoid).
*   **Output**: Probability score (0 = Fake, 1 = Real).

### C. Training Process
*   **Adversarial Loop**: The Generator tries to fool the Discriminator, while the Discriminator tries to correctly classify Real vs. Fake images.
*   **Loss Function**: Binary Cross Entropy Loss (standard GAN loss).

## 4. Workflow Pipeline
1.  **Data Preprocessing**: 
    *   Load raw images.
    *   Resize to 64x64.
    *   Convert to Grayscale.
    *   Normalize pixel values to [-1, 1].
2.  **Model Training**: Train the GAN for ~100 epochs.
3.  **Generation**: Use the trained Generator to create new synthetic signatures.
4.  **Verification Experiment**:
    *   Train a baseline verifier (CNN) on **Real Data Only**.
    *   Train another verifier on **Real + Synthetic Data (Augmented)**.
    *   Compare performance to see if synthetic data helps.

## 5. Evaluation Metrics
*   **Visual Quality**: Do the signatures look real?
*   **FAR (False Acceptance Rate)**: How often a forgery is accepted as genuine.
*   **FRR (False Rejection Rate)**: How often a genuine signature is rejected.
*   **EER (Equal Error Rate)**: The point where FAR = FRR. Lower is better.

## 6. Tech Stack
*   **Language**: Python 3.8+
*   **Framework**: PyTorch (Deep Learning models)
*   **Libraries**: OpenCV (Image proc), NumPy, Matplotlib (Visualization)
*   **UI/Demo**: Gradio / Streamlit (Interactive web interface)

## 7. Future Scope
*   **Conditional GAN (cGAN)**: Generate signatures for a *specific* person ID.
*   **CycleGAN**: Convert offline (scanned) signatures to online (tablet) data.
*   **Higher Resolution**: Scale up to 128x128 or 256x256 images.
