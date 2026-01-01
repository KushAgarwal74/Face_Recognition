# Responsibilities
    # Load images
    # Preprocess (grayscale, resize, normalize, flatten)
    # Build X, y
    # Split train/test

from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# Step 1: convert image to grayscale
def load_image_grayscale(image_path: Path) -> np.ndarray:
    """
    Load an image and convert it to grayscale.
    :param image_path: Description
    :type image_path: Path
    
    Returns: image (H, W) as NumPy array
    :rtype: ndarray
    """

    with Image.open(image_path) as img:
        gray_img = img.convert("L")         # L = luminance (grayscale)
        image_array = np.array(gray_img)

        # This does exactly what PCA/LDA (models Intensity structure not color semantics) expects :
            # Converts RGB → single-channel intensity
            # Uses a standard luminance formula (not a naïve average)
            # Produces a 2D array (H, W)
            # Keeps values as uint8 (0–255)

    return image_array


# This is what we want at this stage
    # images = [img1, img2, img3, ...]   # each img is (H, W)
    # labels = [0, 0, 1, 2, ...]

# Why resizing is mandatory
    # PCA/LDA require fixed-length vectors
    # Different image sizes → invalid data matrix
    # 100 * 100 is ok
    # enough facial detail
    # 10,000 dimensions
    # Eigenfaces still interpretable
    # fast experimentation

    # Later experiment with 
    # 80 * 80
    # 112 * 92 (ORL-style)

# Step 2: Resize and normalize grayscale image 
target_size = (100, 100)  # (width, height)

def resize_and_normalize(image: np.ndarray) -> np.ndarray:
    """
    Resize a grayscale image and normalize pixel values.

    Input:
        iamge: (H, W) uint8 array
    
    Output:
        image: (100, 100) float32 array in range [0, 1]
    """

    # Resize
    # image was a NumPy array (H, W)
    # Resizing is an image-processing operation
    # PIL provides:
    # Interpolation methods (bilinear, bicubic, etc.)
    # Correct geometric handling
    img_pil = Image.fromarray(image)        # converts numpy array to raw pixel data to process it using PIL
    img_resized = img_pil.resize(target_size, Image.BILINEAR)
    
    # Data form after resizing
    # Shape: (100, 100) (fixed)
    # Type: still uint8
    # Range: [0, 255]

    # Normalization - converts pixel values from 0-255 to 0.0-1.0 converts type to flaot32
    # Convert to NumPy
    img_array = np.array(img_resized, dtype=np.float64)

    # Normalize to [0, 1]
    img_array /= 255.0

    # Data form
    # Shape: (100, 100)
    # Type: float32
    # Range: [0.0, 1.0]
    
    # Each pixel now answers:
    # “What fraction of maximum brightness is this?”
    
    # This is numerical conditioning, not visual change.

    return img_array


# Step 3: Flatten normalized image
def flatten_image(image: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D image into a 1D feature vector.

    Input:
        iamge: (H, W) float32 array in range [0, 1]
    
    Output:
        vector: (H*W, ) array
    """

    return image.flatten()


# Step 4: build dataset

DATA_DIR = Path("data/raw")

def build_dataset(): 
    """
    Build feature matrix X and label vector y from data/raw/.

    Return:
        X: (N, D) numpy array
        y: (N, ) numpy array
    """

    X = []
    y = []

    # Sort folders to keep label assignment stable
    subject_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    for label, subject_dir in enumerate(subject_dirs):
        image_paths = sorted(subject_dir.glob("*.png"))

        for img_path in image_paths:
            # 1. Load grayscale image
            image_gray = load_image_grayscale(img_path)

            # 2. Resize and normalize image
            image_resized = resize_and_normalize(image_gray)

            # 3. Flatten image
            image_vector = flatten_image(image_resized)

            # 4. Append
            X.append(image_vector)
            y.append(label)

    return np.array(X), np.array(y)

# Step 5: Split Dataset

def train_test_split_stratified(
    X: np.ndarray,                  # feature matrix Shape: (N, D)
    y: np.ndarray,                  # labels Shape: (N, )   N: no. of images
    test_ratio: float = 0.3,        # 30% images go to test per subject
    seed: int = 42                  # ensures reproducibility same split every run
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Perform image-level stratified train/test split

    Returns:
        X_train, X_test, Y_train, Y_test
    """

    rng = np.random.default_rng(seed)      

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)
    
    train_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        split_point = int(len(indices) * (1-test_ratio))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    # Shuffle final indices (importatant)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )

# X = []
# y = []
# X , y = build_dataset()

# X_train, X_test, y_train, y_test = train_test_split_stratified(
#     X, y, test_ratio=0.3
# )

# print("Train samples:", len(X_train))
# print("Test samples:", len(X_test))

# print(X_train.shape, X_test.shape)

# from collections import Counter

# print("Train labels:", Counter(y_train))
# print("Test labels:", Counter(y_test))




# Test if grayscale image is resized and normalized correctly
# img_gray = load_image_grayscale(Path("data/raw/subject_1/0.png"))
# img_ready = resize_and_normalize(img_gray)
# img_vector = flatten_image(img_ready)

# print(img_gray)
# print("After Grayscale:", img_gray.shape)        # give the (H, W)
# print(img_gray.dtype)        # uint8

# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")
# plt.show()

# print(img_ready)
# print("After resizing and normalize:", img_ready.shape)        # give the (H, W)
# print(img_ready.dtype)        # uint8
# print(img_ready.min())        # >= 0.0 
# print(img_ready.max())        # <= 1.0

# plt.imshow(img_ready, cmap="gray")
# plt.axis("off")
# plt.show()

# print(img_vector)
# print("After Flattening:", img_vector.shape)        # give the (H, W)
# print(img_vector.dtype)        # uint8
# print(img_vector.min())        # 0.0 (or close)
# print(img_vector.max())        # 1.0 (or close)
