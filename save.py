import numpy as np
import cv2
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_PATH = 'tissuemnist_224.npz' # The cleaned grayscale file
OUTPUT_DIR = 'outputs_tissue'
FINAL_PATH = os.path.join(OUTPUT_DIR, 'tissue_64_enhanced.npz')

# Target Size: 64x64 is optimal for TissueMNIST (source is 28x28)
TARGET_SIZE = 64 

# === LOAD DATA ===
print(f"Loading {INPUT_PATH}...")
data = np.load(INPUT_PATH)
# Load all splits
train_img = data['train_images']
val_img = data['val_images']
test_img = data['test_images']

# Load labels & weights
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']
class_weights = data['class_weights']

print(f"Original Shape: {train_img.shape} (Mean ~{data['unified_mean'][0]:.3f})")

# === PROCESSING PIPELINE ===
# 1. Resize (224 -> 64)
# 2. CLAHE (Contrast Enhancement)

def process_batch(images, desc):
    processed = []
    # Create CLAHE object (Clip limit 2.0 is standard for medical)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for img in tqdm(images, desc=desc):
        # Resize first (faster)
        # Interpolation AREA is best for downsampling (removes noise)
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        # Apply CLAHE
        enhanced = clahe.apply(resized)
        processed.append(enhanced)
        
    return np.array(processed, dtype=np.uint8)

print("\n--- Starting Enhancement Pipeline ---")
train_proc = process_batch(train_img, "Processing Train")
val_proc   = process_batch(val_img,   "Processing Val  ")
test_proc  = process_batch(test_img,  "Processing Test ")

print(f"\nNew Shape: {train_proc.shape}")

# === RECALCULATE STATS ===
print("Recalculating Mean/Std on enhanced data...")
all_pixels = np.concatenate([train_proc, val_proc, test_proc], axis=0)
pixel_mean = np.mean(all_pixels) / 255.0
pixel_std = np.std(all_pixels) / 255.0

print(f"New Mean: {pixel_mean:.4f} (Was ~0.10)")
print(f"New Std:  {pixel_std:.4f}")

# === VISUALIZATION CHECK ===
# Compare Original vs Processed
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
indices = np.random.choice(len(train_img), 5)

for i, idx in enumerate(indices):
    # Original (Dark/Blurry)
    axes[0, i].imshow(train_img[idx], cmap='gray')
    axes[0, i].set_title("Original (224x224)")
    axes[0, i].axis('off')
    
    # Processed (Sharp/Contrast)
    axes[1, i].imshow(train_proc[idx], cmap='gray')
    axes[1, i].set_title("Final (64x64 + CLAHE)")
    axes[1, i].axis('off')

plt.savefig(os.path.join(OUTPUT_DIR, "final_comparison.png"))
print("\nSaved comparison image to final_comparison.png. Check this!")

# === SAVE FINAL ===
print(f"\nSaving to {FINAL_PATH}...")
np.savez_compressed(
    FINAL_PATH,
    train_images=train_proc, train_labels=train_labels,
    val_images=val_proc, val_labels=val_labels,
    test_images=test_proc, test_labels=test_labels,
    unified_mean=np.array([pixel_mean]*3), # Store as 3-channel ready
    unified_std=np.array([pixel_std]*3),
    class_weights=class_weights
)

print("✅ SUCCESS! Upload 'tissue_64_enhanced.npz' to your Drive.")