import numpy as np
import matplotlib.pyplot as plt
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm
from collections import Counter

# === CONFIG ===
INPUT_PATH = 'tissuemnist_224.npz' # Load the one we just saved
SAMPLE_SIZE = 5000 # Analyze a subset for speed (or use len(train_images) for full)

# === LOAD DATA ===
print(f"Loading {INPUT_PATH}...")
data = np.load(INPUT_PATH)
images = data['train_images'] # We analyze training data primarily
labels = data['train_labels']

# Ensure images are uint8 (0-255)
if images.dtype != np.uint8:
    images = (images * 255).astype(np.uint8)

print(f"Loaded {len(images)} images. Analyzing random subset of {SAMPLE_SIZE}...")
indices = np.random.choice(len(images), SAMPLE_SIZE, replace=False)
sample_imgs = images[indices]

# ==========================================
# 1. BLURRINESS CHECK (Laplacian Variance)
# ==========================================
print("\n--- 1. Blurriness Analysis (Laplacian Var) ---")
variances = []
for img in tqdm(sample_imgs, desc="Calculating Blur"):
    # Laplacian Variance: High = Sharp, Low = Blurry
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    variances.append(var)

avg_var = np.mean(variances)
min_var = np.min(variances)
max_var = np.max(variances)

print(f"Average Sharpness Score: {avg_var:.2f}")
print(f"Min (Blurriest): {min_var:.2f} | Max (Sharpest): {max_var:.2f}")

if avg_var < 100:
    print("⚠️ VERDICT: Images are VERY BLURRY. Downsizing to 64x64 or 128x128 might IMPROVE accuracy.")
else:
    print("✅ VERDICT: Images have decent texture detail.")

# ==========================================
# 2. CONTRAST CHECK (Histogram)
# ==========================================
print("\n--- 2. Contrast Analysis ---")
# Flatten all pixels in the sample
all_pixels = sample_imgs.flatten()

plt.figure(figsize=(10, 4))
plt.hist(all_pixels, bins=50, color='gray', alpha=0.7, label='Original')
plt.title("Pixel Intensity Distribution (Original)")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Count")
plt.axvline(x=np.mean(all_pixels), color='r', linestyle='--', label=f'Mean: {np.mean(all_pixels):.1f}')
plt.legend()
plt.savefig("outputs_tissue/pixel_histogram.png")
plt.show() # If in notebook

# Test CLAHE (Contrast Enhancement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_sample = [clahe.apply(img) for img in sample_imgs[:5]]

# Show Before/After
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # Original
    axes[0, i].imshow(sample_imgs[i], cmap='gray')
    axes[0, i].set_title(f"Original (Var: {variances[i]:.0f})")
    axes[0, i].axis('off')
    
    # Enhanced
    axes[1, i].imshow(enhanced_sample[i], cmap='gray')
    axes[1, i].set_title("With CLAHE Fix")
    axes[1, i].axis('off')
plt.suptitle("Impact of Contrast Enhancement")
plt.savefig("outputs_tissue/clahe_comparison.png")
print("Saved comparison to outputs_tissue/clahe_comparison.png")

# ==========================================
# 3. NEAR-DUPLICATE CHECK (pHash)
# ==========================================
print("\n--- 3. Near-Duplicate Analysis (pHash) ---")
# We check a smaller subset for pHash because it's O(N^2) complexity to compare all
PHASH_SUBSET = 1000 
phash_imgs = images[:PHASH_SUBSET]
hashes = []

print(f"Hashing first {PHASH_SUBSET} images...")
for img in phash_imgs:
    pil_img = Image.fromarray(img)
    hashes.append(imagehash.phash(pil_img))

# Brute force comparison
duplicates = 0
threshold = 2 # Hamming distance (0 = exact, <5 = very similar)

print("Comparing hashes...")
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        if hashes[i] - hashes[j] <= threshold:
            duplicates += 1
            if duplicates < 5: # Print first few matches
                print(f"  Found near-duplicate: Img {i} vs Img {j} (Dist: {hashes[i] - hashes[j]})")

print(f"Found {duplicates} near-duplicates in a sample of {PHASH_SUBSET} images.")
if duplicates > 0:
    print("⚠️ VERDICT: pHash found duplicates that MD5 missed! We need a stronger cleaning filter.")
else:
    print("✅ VERDICT: No near-duplicates found in this subset.")