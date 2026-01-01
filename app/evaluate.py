import numpy as np
from collections import Counter
from typing import Tuple
# import pandas as pd

from app.infer import predict_nearest_neighbor_multi, predict_nearest_neighbor_single
from core.lda import fit_lda
from app.dataset import train_test_split_stratified

# -----------------------
# 1️⃣ Compute Accuracy
# -----------------------
def compute_accuracy(
    y_true : np.ndarray,
    y_pred : np.ndarray
) -> float:
    """
    Compute Classification Accuracy
    """

    return float(np.mean(y_true == y_pred))

# -----------------------
# 2️⃣ Confusion Matrix
# -----------------------
def confusion_matrix(y_true: np.ndarray , y_pred: np.ndarray)-> np.ndarray:
    """
    Build Confusion Matrix.
    Rows = True Labels
    Columns = Predicted Labels
    """

    num_classes = len(np.unique(y_true))
    cm = np.zeros((num_classes, num_classes), dtype = int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm

# -------------------------------
# 3️⃣ Misclassification Indices
# -------------------------------
def misclassified_indices(
        y_true : np.ndarray,
        y_pred : np.ndarray
)-> np.ndarray:
    """
    Return array of indices where predictions != true value
    """
    return np.where(y_pred != y_true)[0]

# -------------------
# 4️⃣ Compute Model
# -------------------
def evaluate_model(
    X_test_PCA: np.ndarray,
    y_test: np.ndarray,
    X_train_PCA: np.ndarray,
    y_train: np.ndarray,
    distance: str = "cosine"
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run NN on test set and compute metrics.

    Returns:
        accuracy
        confusion_matrix
        misclassified_indices
    """

    # Predict
    y_pred = predict_nearest_neighbor_multi(
        X_test_PCA,
        X_train_PCA,
        y_train,
        distance=distance
    )

    # Metrics
    acc = compute_accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    mis_idx = misclassified_indices(y_test, y_pred)

    return acc, cm, mis_idx

# -----------------------------------------------------------
# 5️⃣ Compare Accuracy PCA vs PCA + LDA
# -----------------------------------------------------------
def compare_pca_vs_lda(
    X_train_PCA,
    X_test_PCA,
    y_train,
    y_test,
    distance: str = "cosine"
)-> tuple[float, float]:
    
    # PCA Only
    y_pred_pca = predict_nearest_neighbor_multi(
        X_test_PCA,
        X_train_PCA,
        y_train,
        distance
    )
    acc_pca = compute_accuracy(y_test, y_pred_pca)

    # PCA + LDA
    num_classes = len(np.unique(y_train))
    w_lda = fit_lda(X_train_PCA, y_train, num_classes -1)

    X_train_lda = X_train_PCA @ w_lda
    X_test_lda = X_test_PCA @ w_lda
    
    y_pred_lda  = predict_nearest_neighbor_multi(
        X_test_lda,
        X_train_lda,
        y_train,
        distance
    )
    acc_lda = compute_accuracy(y_test, y_pred_lda)
    
    return acc_pca, acc_lda

# -----------------------------------------------------------
# 6️⃣ PCA Accuracy vs k plot using cosine and euclidean distance
# -----------------------------------------------------------
def accuracy_k_pca_plot(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    mean_face : np.ndarray,
    eigenfaces : np.ndarray,
    k_max : int,
    distance = ("euclidean", "cosine")
) -> tuple[list, list, list]: 
    
    k_values = list(range(2, k_max))
    acc_euclid = []
    acc_cosine = []
    
    for i in k_values:
        eigenfaces_k = eigenfaces[:, :i]

        # Porject
        X_train_PCA = (X_train - mean_face) @ eigenfaces_k
        X_true_proj = (X_test - mean_face) @ eigenfaces_k

        # Predict
        y_pred_e = predict_nearest_neighbor_multi(X_true_proj, X_train_PCA, y_train, distance[0])
        y_pred_c = predict_nearest_neighbor_multi(X_true_proj, X_train_PCA, y_train, distance[1])

        # Accuracy
        acc_euclid.append(compute_accuracy(y_test, y_pred_e))
        acc_cosine.append(compute_accuracy(y_test, y_pred_c))

    return k_values, acc_euclid, acc_cosine

# -----------------------------------------------------------
# 7️⃣ LDA Accuracy vs k plot using cosine and euclidean distance
# -----------------------------------------------------------
def accuracy_k_lda_plot(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_lda_max: int,
    distance = ("euclidean", "cosine")
) -> tuple[list, list, list]:
    """
    Accuracy vs number of LDA components.

    Parameters
    ----------
    X_train_pca : (N_train, k_pca)
    X_test_pca  : (N_test, k_pca)
    y_train     : (N_train,)
    y_test      : (N_test,)
    k_lda_max   : int (<= num_classes - 1)

    Returns
    -------
    k_values : list[int]
    accuracies_euclidean : list[float]
    accuracies_cosine : list[float]
    """

    k_values = list(range(1, k_lda_max + 1))
    accuracies_c = []
    accuracies_e = []

    for k_lda in k_values:
        # Fit LDA
        W_lda = fit_lda(X_train_pca, y_train, k_lda)

        # Project
        X_train_lda = X_train_pca @ W_lda
        X_test_lda  = X_test_pca @ W_lda

        # Predict
        y_pred_c = predict_nearest_neighbor_multi(
            X_test_lda,
            X_train_lda,
            y_train,
            distance=distance[0]
        )
        y_pred_e = predict_nearest_neighbor_multi(
            X_test_lda,
            X_train_lda,
            y_train,
            distance=distance[1]
        )

        # Accuracy
        accuracies_c.append(compute_accuracy(y_test, y_pred_c))
        accuracies_e.append(compute_accuracy(y_test, y_pred_e))

    return k_values, accuracies_e, accuracies_c

def select_klda_via_validation(
    X_train_PCA : np.ndarray,
    y_train : np.ndarray,
    k_lda_max : int,
    test_size : float,
    random_state : int,
    distance : str =  "cosine"
) -> int:
    """
    Determine the best k_lda by splitting dataset on X_train_PCA

    Paramerts
    ----------
    X_train_PCA  : (N_train, k_pca),
    y_train      : (N_train, ),
    k_lda        : int,
   
    Returns
    --------
    best_k_lda   : int
    """

    X_train_sub, X_val, y_train_sub, y_val = train_test_split_stratified(X_train_PCA,
                                                                         y_train,
                                                                         test_size,
                                                                         random_state)
    
    k_val_lda, acc_euclid, acc_cosine = accuracy_k_lda_plot(X_train_sub,
                                                            X_val,
                                                            y_train_sub,
                                                            y_val,
                                                            k_lda_max)
    
    if distance == "cosine":
        best_k_lda = k_val_lda[np.argmax(acc_cosine)]
    else:
        best_k_lda = k_val_lda[np.argmax(acc_euclid)]

    return best_k_lda