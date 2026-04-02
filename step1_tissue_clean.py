import numpy as np
import os
import json
from collections import Counter
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_PATH = 'tissuemnist_224.npz' 
OUTPUT_DIR = 'outputs_tissue'
FINAL_NPZ_PATH = os.path.join(OUTPUT_DIR, 'final_tissuemnist_224.npz')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {
    0: 'Kidney Cortex', 1: 'Kidney Medulla', 2: 'Kidney Pelvis',
    3: 'Lung Adenocarcinoma', 4: 'Lung Squamous Cell Carcinoma',
    5: 'Pancreas Ductal Adenocarcinoma', 6: 'Pancreas Neuroendocrine Tumor',
    7: 'Pancreas Solid Pseudopapillary Tumor'
}

# === LOAD DATA ===
print(f"Loading {INPUT_PATH}...")
data = np.load(INPUT_PATH)
train_images, train_labels = data['train_images'], data['train_labels'].flatten()
val_images, val_labels = data['val_images'], data['val_labels'].flatten()
test_images, test_labels = data['test_images'], data['test_labels'].flatten()

# === EDA: CLASS DISTRIBUTION ===
print("\n--- EDA: Class Distribution ---")
train_counts = Counter(train_labels)
classes = sorted(train_counts.keys())
total_samples = sum(train_counts.values())
num_classes = len(classes)

# Inverse frequency weights
class_weights = {int(cls): total_samples / (num_classes * train_counts[cls]) for cls in classes}
print(f"Cleaned Train Counts: {dict(train_counts)}")

# === STEP 2: CHUNKED STATS CALCULATION (Memory Safe) ===
print("\n--- Step 2: Chunked Global Stats (PC Optimized) ---")

def calculate_stats_chunked(splits, chunk_size=1000):
    """Calculates mean and std by processing small batches to save RAM."""
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    
    for split in splits:
        num_imgs = split.shape[0]
        for i in tqdm(range(0, num_imgs, chunk_size), desc="Processing Chunks"):
            chunk = split[i : i + chunk_size].astype(np.float32) / 255.0
            total_sum += np.sum(chunk)
            total_sq_sum += np.sum(np.square(chunk))
            total_pixels += chunk.size
            
    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(max(0, var))
    return mean, std

pixel_mean, pixel_std = calculate_stats_chunked([train_images, val_images, test_images])

final_mean = [float(pixel_mean)] * 3
final_std = [float(pixel_std)] * 3

print(f"\nUnified Mean: {final_mean}")
print(f"Unified Std:  {final_std}")

# === STEP 3: SAVING ===
print(f"\n--- Step 3: Saving Processed Data ---")
# Keep as uint8 to save disk space
np.savez_compressed(
    FINAL_NPZ_PATH,
    train_images=train_images, train_labels=train_labels,
    val_images=val_images, val_labels=val_labels,
    test_images=test_images, test_labels=test_labels,
    unified_mean=np.array(final_mean),
    unified_std=np.array(final_std),
    class_weights=np.array([class_weights[i] for i in range(num_classes)])
)

metadata = {
    "class_names": CLASS_NAMES,
    "class_weights": class_weights,
    "mean": final_mean,
    "std": final_std,
    "final_counts": {
        "train": len(train_images),
        "val": len(val_images),
        "test": len(test_images)
    }
}

with open(os.path.join(OUTPUT_DIR, 'tissue_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print("DONE. Processed Grayscale dataset saved successfully on local PC.")