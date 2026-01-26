"""
Preprocessing Utilities
=======================
Functions for preprocessing signature images.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Reads, binarizes, crops, and resizes a signature image.
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired output size (width, height).
        
    Returns:
        numpy.ndarray: Preprocessed image normalized to [-1, 1], or None if failed.
    """
    try:
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Denoise
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        # Binarize using adaptive thresholding (to handle uneven lighting)
        # Invert so signature is white on black (standard for some processing) or keep black on white.
        # Usually GANs for MNIST-like data use White digit on Black background [0, 1] range.
        # Signatures are usually black ink on white paper.
        # Let's standardize to Black Ink on White Background for visual coherence, 
        # but for GAN training, centering data around 0 is key.
        # We will return [-1, 1] where -1 is black (ink) and 1 is white (background) or vice versa.
        # Let's stick to standard: -1 (black), 1 (white).
        
        # Binary threshold (OTSU is good for bimodal, Adaptive for varying lighting)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find bounding box of signature (non-zero pixels)
        points = cv2.findNonZero(thresh)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            # Crop
            crop = img[y:y+h, x:x+w]
        else:
            crop = img # Fallback if empty
            
        # Resize to target size with padding to preserve aspect ratio
        h_c, w_c = crop.shape
        ratio = min(target_size[0] / w_c, target_size[1] / h_c)
        new_w = int(w_c * ratio)
        new_h = int(h_c * ratio)
        
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create blank canvas (white background = 255)
        canvas = np.full(target_size, 255, dtype=np.uint8)
        
        # Center the image
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize to [-1, 1]
        # 0 -> -1 (Black), 255 -> 1 (White)
        normalized = (canvas.astype(np.float32) / 127.5) - 1.0
        
        return normalized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess Signature Images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=64, help="Target image size (square)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    count = 0
    
    print(f"Processing images from {input_path} to {output_path}...")
    
    for file_path in input_path.rglob('*'):
        if file_path.suffix.lower() in supported_exts:
            processed = preprocess_image(str(file_path), target_size=(args.size, args.size))
            
            if processed is not None:
                # Save as PNG. Need to convert back to [0, 255] uint8 for saving
                saved_img = ((processed + 1.0) * 127.5).astype(np.uint8)
                save_name = output_path / file_path.name
                
                # Ensure unique name if flattened
                if save_name.exists():
                    save_name = output_path / f"{file_path.stem}_{count}{file_path.suffix}"
                
                cv2.imwrite(str(save_name), saved_img)
                count += 1
                
    print(f"Successfully processed {count} images.")

if __name__ == '__main__':
    main()
