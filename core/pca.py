import numpy as np
from typing import Tuple

def fit_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA using SVD

    Parameters 
    -----------
    X : ndarray of shape (N, D)
        Training data (flattened images)
    k : int
        Number of principal component

    Returns
    --------
    mean_face : ndarray of shape (D,)
    eigenfaces_k : ndarray of shape (D, k)
    """

    # Mean Face
    mean_face = np.mean(X, axis=0)

    # Centered Data
    X_train_centered = X - mean_face

    # SVD-based PCA (numerically stable)
    U, s, Vt = np.linalg.svd(X_train_centered, full_matrices= False)

    # eigenfaces (right singular vector)
    eigenfaces = Vt.T            #(D, N)

    # Select top-k
    eigenfaces_k = eigenfaces[:, :k]

    return mean_face, eigenfaces, eigenfaces_k

def project(
    X : np.ndarray,
    mean_face : np.ndarray,
    eigenfaces_k : np.ndarray
) -> np.ndarray:
    """
    Project data into PCA space
    """
    return (X - mean_face) @ eigenfaces_k

def reconstruct(
    X_proj : np.ndarray,
    mean_face : np.ndarray,
    eigenfaces_k : np.ndarray
) -> np.ndarray:
    """
    Reconstruct data from PCA space
    """
    return mean_face + X_proj @ eigenfaces_k.T
