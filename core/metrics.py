import numpy as np

#____________________
# Distance Functions
#____________________

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / denom
