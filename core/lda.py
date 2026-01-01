# What PCA cares about
#     Lighting variation
#     Pose variation
#     Expression variation
# What PCA does NOT care about
#     Class labels
#     Identity boundaries

# Why PCA Fails for Classification (Key Insight)
# High variance ≠ high discrimination.
# Example:
#     Lighting changes create huge variance
#     Identity differences may be subtle
# So PCA may choose directions that:
#     Separate bright vs dark images
#     NOT person A vs person B
# This is why you saw identity confusion.

# LDA asks a different question:
#     “Which directions best separate known classes?”
# Instead of maximizing variance, LDA maximizes class separability.

# ‼️ LDA is dones when these questions are answered
# | Question                                | You should have         |
# | --------------------------------------- | ----------------------- |
# | Does Fisherfaces outperform Eigenfaces? | Accuracy comparison     |
# | How many LDA components matter?         | Accuracy vs `k_lda`     |
# | What does LDA space look like?          | 2D / 3D visualization   |
# | Does distance metric still matter?      | Cosine vs Euclid in LDA |
# | Is training/test separation respected?  | Clean pipeline          |



import numpy as np
from typing import Dict

def compute_class_mean(X_train_PCA: np.ndarray, y_train: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute mean vector for each class.

    Parameters
    -----------
    X_train_PCA: ndarray of shape (N, k)
       Feature matrix (e.g., PCA-projected training data)
    y: ndarrya of shape (N, )
       Class labels corresponding to X_train_PCA
    """
    classes = np.unique(y_train)
    return {c : X_train_PCA[y_train == c].mean(axis=0) for c in classes}

def compute_within_class_scatter(X_train_PCA: np.ndarray, y_train: np.ndarray, class_means: Dict[int, np.ndarray])-> np.ndarray:
    """
    Compute within-class scatter matrix Sw.

    Parameters
    -----------
    X_train_PCA: ndarray of shape (N, k)
       Feature matrix (e.g., PCA-projected training data)
    y: ndarrya of shape (N, )
       Class labels corresponding to X_train_PCA
    class_means : dict[int, ndarray of shape (k,)]
        Precomputed class means
    """

    num_features = X_train_PCA.shape[1]
    Sw = np.zeros((num_features, num_features))

    for labels, mean_vector in class_means.items():
        X_class = X_train_PCA[y_train == labels]
        deviations = X_class - mean_vector
        Sw += deviations.T @ deviations
    
    return Sw

def compute_between_class_scatter(X_train_PCA: np.ndarray, y_train: np.ndarray, class_means: Dict[int, np.ndarray])-> np.ndarray:
    """
    Compute between-class scatter matrix Sw.

    Parameters
    -----------
    X_train_PCA: ndarray of shape (N, k)
       Feature matrix (e.g., PCA-projected training data)
    y: ndarrya of shape (N, )
       Class labels corresponding to X_train_PCA
    class_means : dict[int, ndarray of shape (k,)]
        Precomputed class means
    """

    overall_mean = np.mean(X_train_PCA, axis =0)
    num_features = X_train_PCA.shape[1]

    Sb = np.zeros((num_features, num_features))

    for label, mean_vector in class_means.items():
        number_of_class_samples = np.sum(y_train == label)
        mean_diff = (mean_vector - overall_mean).reshape(-1, 1)
        Sb += number_of_class_samples * (mean_diff @ mean_diff.T)

    return Sb

def fit_lda(
    X_train_PCA: np.ndarray,
    y_train: np.ndarray,
    num_components: int
) -> np.ndarray:
    """
    Fit Linear Discriminant Analysis (LDA).

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Training features after PCA projection
    y : ndarray of shape (N,)
        Training class labels
    num_components : int
        Number of LDA components (<= number of classes - 1)

    Returns
    -------
    W_lda : ndarray of shape (D, num_components)
        LDA projection matrix (Fisherfaces)
    """

    # Step 1: Compute class statistics
    class_means = compute_class_mean(X_train_PCA, y_train)

    # Step 2: Compute scatter matrices
    Sw = compute_within_class_scatter(X_train_PCA, y_train, class_means)
    Sb = compute_between_class_scatter(X_train_PCA, y_train, class_means)

    # Step 3: Solve generalized eigenvalue problem
    # Sb * w = lambda * Sw * w
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)

    # Step 4: Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigvals.real)[::-1]
    W_lda = eigvecs[:, sorted_indices[:num_components]].real

    return W_lda

