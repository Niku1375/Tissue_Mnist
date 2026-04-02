import numpy as np
import cv2
import os
from tqdm import tqdm
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your ORIGINAL downloaded file
INPUT_PATH = 'tissuemnist_224.npz' 

# 2. Output settings
OUTPUT_DIR = 'outputs_tissue'
FINAL_FILENAME = 'tissue_64_enhanced.npz'
FINAL_PATH = os.path.join(OUTPUT_DIR, FINAL_FILENAME)

# 3. Processing Parameters (Based on Analysis)
TARGET_SIZE = 64   # Downsizing to 64x64 removes interpolation blur
CLAHE_CLIP = 3.0   # 3.0 is optimal for dark medical images (Standard is 2.0)
CLAHE_GRID = (8,8)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def process_batch(images, desc):
    """
    Applies Resize and CLAHE to a batch of images.
    """
    processed = []
    # Initialize CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    
    for img in tqdm(images, desc=desc):
        # Step 1: Ensure Grayscale (Robustness check)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Step 2: Resize (Area interpolation is best for shrinking)
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        # Step 3: Apply CLAHE
        # Note: CLAHE expects uint8 (0-255) input
        enhanced = clahe.apply(resized)
        
        processed.append(enhanced)
        
    return np.array(processed, dtype=np.uint8)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. LOAD DATA
    print(f"Loading original data from {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find input file at {INPUT_PATH}")
        
    data = np.load(INPUT_PATH)
    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']
    
    # 2. PROCESS IMAGES
    print(f"\n--- Starting Processing Pipeline (Resize {TARGET_SIZE}x{TARGET_SIZE} + CLAHE) ---")
    train_proc = process_batch(data['train_images'], "Processing Train")
    val_proc   = process_batch(data['val_images'],   "Processing Val  ")
    test_proc  = process_batch(data['test_images'],  "Processing Test ")

    # 3. CALCULATE METADATA
    print("\n--- Calculating Dataset Statistics ---")
    
    # A. Class Weights (Inverse Frequency)
    # This is crucial for your Loss Function later
    counts = Counter(train_labels.flatten())
    total_samples = len(train_labels)
    num_classes = len(counts)
    
    # Sort keys to ensure weight list matches class index 0, 1, 2...
    sorted_keys = sorted(counts.keys())
    class_weights_dict = {k: total_samples / (num_classes * v) for k, v in counts.items()}
    weights_list = [class_weights_dict[k] for k in sorted_keys]
    
    print(f"Class Counts: {dict(counts)}")
    print(f"Computed Class Weights: {[round(w, 2) for w in weights_list]}")

    # B. Global Mean & Std (For Normalization)
    # We combine all splits to get a global stat, normalizing to 0-1 range first
    print("Computing pixel Mean/Std...")
    all_pixels = np.concatenate([train_proc, val_proc, test_proc], axis=0)
    # Convert to float32 for precision during calculation
    pixel_mean = np.mean(all_pixels) / 255.0
    pixel_std = np.std(all_pixels) / 255.0
    
    print(f"New Global Mean: {pixel_mean:.4f}")
    print(f"New Global Std:  {pixel_std:.4f}")

    # 4. SAVE FINAL DATASET
    print(f"\nSaving final processed data to {FINAL_PATH}...")
    np.savez_compressed(
        FINAL_PATH,
        # Image Data (uint8 to save space)
        train_images=train_proc,
        val_images=val_proc,
        test_images=test_proc,
        # Labels
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        # Metadata (Saved as 3-channel values for compatibility with RGB models)
        unified_mean=np.array([pixel_mean] * 3), 
        unified_std=np.array([pixel_std] * 3),
        class_weights=np.array(weights_list)
    )

    print(f"\n✅ SUCCESS! Dataset saved: {FINAL_PATH}")
    print(f"   Size: 64x64 | Format: uint8 Grayscale | Enhanced: CLAHE 3.0")
    print("   You can now use this file for modeling.")