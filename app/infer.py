import numpy as np
from pathlib import Path
from PIL import Image

from app.dataset import(load_image_grayscale, resize_and_normalize,flatten_image)
from core.metrics import euclidean_distance, cosine_distance

#_____________________________
# Nearest Neighbor Prediction
#_____________________________

DISTANCE_FUNCTIONS = {"euclidean": euclidean_distance, "cosine": cosine_distance}

def predict_nearest_neighbor_single(
    X_query: np.ndarray,
    X_train_PCA: np.ndarray,
    y_train: np.ndarray,
    distance = "euclidean"
) -> int:
    """
    Predict identity for a single projected face vector.
    """
    distance_fn = DISTANCE_FUNCTIONS[distance]
    distances = [distance_fn(X_query, x_train) for x_train in X_train_PCA]
    neareset_idx = np.argmin(distances)          # return index of min distance
    return int(y_train[neareset_idx])

def predict_nearest_neighbor_multi(
    X_query: np.ndarray,
    X_train_PCA: np.ndarray,
    y_train: np.ndarray,
    distance = "euclidean"
) -> np.ndarray:

    return np.array([
        predict_nearest_neighbor_single(x, X_train_PCA, y_train, distance)
        for x in X_query
    ])

#_____________________________
# Full Inference Pipeline
#_____________________________

def infer_single_image(
    image_path : Path,
    mean_face : np.ndarray,
    eigenfaces_k : np.ndarray,
    X_train_PCA: np.ndarray,
    y_train : np.ndarray,
    distance = ["euclidean","cosine"]
) -> tuple[int, int]:

    """ 
    Predict identity for a single input image
    """

    # 1. Load & Preprocess
    img = load_image_grayscale(image_path)
    img = resize_and_normalize(img)
    img = flatten_image(img)

    # 2. Center using training mean
    img_centered = img - mean_face

    # 3. Project into eigenface space
    img_proj = img_centered @ eigenfaces_k

    # 4. Nearest Neighbor Classification
    predicted_label_euclid = predict_nearest_neighbor_single(img_proj, X_train_PCA, y_train, distance[0])
    predicted_label_cosine = predict_nearest_neighbor_single(img_proj, X_train_PCA, y_train, distance[1])

    return predicted_label_euclid, predicted_label_cosine
