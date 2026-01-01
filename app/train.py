# from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

from app.dataset import build_dataset, train_test_split_stratified
from core.pca import fit_pca, project, reconstruct
from core.lda import fit_lda
from app.evaluate import (evaluate_model, compare_pca_vs_lda, select_klda_via_validation,
                          accuracy_k_pca_plot, accuracy_k_lda_plot)

def load_config(path ="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    # -----------------------
    # 1. Load dataset
    # -----------------------
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, 
        test_ratio = cfg["split"]["test_ratio"], 
        seed = cfg["split"]["seed"]
    )

    # -----------------------
    # 2. PCA (Eigenfaces)
    # -----------------------
    if not cfg["pca"]["enabled"]:
        raise ValueError("PCA must be enabled before LDA.")
    
    k_pca = cfg["pca"]["k"]
    num_classes = len(np.unique(y_train))
    max_pca_dim = X_train.shape[0] - num_classes

    if k_pca > max_pca_dim:
        raise ValueError(f"PCA k = {k_pca} too large. Max allowed = {max_pca_dim}")

    mean_face, eigenfaces, eigenfaces_k = fit_pca(X_train, k_pca)

    # Image Projections along PCA faces in direction of maximum variance
    X_train_PCA = project(X_train, mean_face, eigenfaces_k)
    X_test_PCA = project(X_test, mean_face, eigenfaces_k)

    # -----------------------
    # 3. LDA (Fisherfaces)
    # -----------------------
    if cfg["lda"]["enabled"]:
        
        if cfg["lda"]["n_components"] == "auto":
            if cfg["lda"]["selection"]["enabled"]:
                k_lda = select_klda_via_validation(X_train_PCA,y_train, num_classes-1, 
                                                   cfg["split"]["val_ratio"],
                                                   cfg["split"]["seed"],
                                                   cfg["classifier"]["distance"]
                                                   ) 
            else:
                k_lda = num_classes - 1
        else:
            k_lda = min(cfg["lda"]["n_components"], num_classes - 1)
        
        if k_lda <= 0:
            raise ValueError("LDA requires at least 2 classes.")
        
        # LDA Faces and Projections along PCA faces in direction of maximum class separability
        LDA_faces = fit_lda(X_train_PCA, y_train, k_lda)

        X_train_feat = X_train_PCA @ LDA_faces
        X_test_feat  = X_test_PCA @ LDA_faces

        feature_space = "PCA + LDA"

    else:
        # Fallback: PCA only
        X_train_feat = X_train_PCA
        X_test_feat  = X_test_PCA
        feature_space = "PCA only"


    # Reconstruction
    if cfg.get("debug", False):
        reconstruct_img = reconstruct(X_train_PCA, mean_face, eigenfaces_k)

    # -----------------------
    # 4. Classification & Evaluation
    # -----------------------
    acc, cm, mis_idx = evaluate_model(
        X_test_feat,
        y_test, 
        X_train_feat,
        y_train, 
        distance= cfg["classifier"]["distance"])

    # -----------------------
    # 5. Results
    # -----------------------
    print(f"\nFeature space   : {feature_space}")
    print(f"Test accuracy  : {acc * 100:.2f}%")
    print("Confusion matrix:")
    print(cm)
    print(f"Misclassified samples: {len(mis_idx)}")

    acc_pca, acc_lda = compare_pca_vs_lda(X_train_PCA, X_test_PCA, y_train, y_test,
                                           distance = cfg["classifier"]["distance"])

    print(f"PCA accuracy        :{acc_pca * 100: 2f}%")
    print(f"PCA + LDA accuracy  :{acc_lda * 100: 2f}%")

    k_vals_pca, acc_vals_e_pca, acc_vals_c_pca = accuracy_k_pca_plot(X_train,
                                                                     X_test,
                                                                     y_train,
                                                                     y_test,
                                                                     mean_face,
                                                                     eigenfaces,
                                                                     k_pca)

    plt.figure(figsize=(5,3))
    plt.plot(k_vals_pca, acc_vals_e_pca, acc_vals_c_pca, marker = "o")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Accuracy ('Euclid' vs 'cosine')")
    plt.title("Accuracy vs PCA dimensions")
    plt.grid(True)
    plt.show()

    k_vals_lda, acc_vals_e_lda, acc_vals_c_lda = accuracy_k_lda_plot(X_train_PCA,
                                                                     X_test_PCA,
                                                                     y_train,
                                                                     y_test,
                                                                     num_classes-1)

    plt.figure(figsize=(7,4))

    plt.plot(k_vals_lda, acc_vals_e_lda, marker="o", label="Euclidean")
    plt.plot(k_vals_lda, acc_vals_c_lda, marker="o", label="Cosine")

    plt.xlabel("Number of LDA components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs LDA dimensions")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

# --------------------------
# ‼️ Detailed Math of PCA 
# --------------------------

# mean_face = np.mean(X_train, axis=0)

# X_train_centered = X_train - mean_face

# N, D = X_train_centered.shape
# image_shape = (100, 100)

# # ---------------------------------------------------------
# # 1️⃣ Covariance-based PCA (image space trick)
# # ---------------------------------------------------------

# # Covariance in image space
# C = (X_train_centered @ X_train_centered.T) / (N - 1)

# # Eigen-decomposition
# eigvals, eigvecs = np.linalg.eigh(C)

# # Sort descending
# idx = np.argsort(eigvals)[::-1]
# eigvals = eigvals[idx]
# eigvecs = eigvecs[:, idx]

# # Filter small eigenvalues (relative threshold)
# valid = eigvals > (1e-3 * eigvals[0])
# eigvals = eigvals[valid]
# eigvecs = eigvecs[:, valid]

# # Map to pixel space → eigenfaces
# eigenfaces_cov = X_train_centered.T @ eigvecs   # shape (D, K)

# # Normalize eigenfaces
# eigenfaces_cov = eigenfaces_cov / np.linalg.norm(
#     eigenfaces_cov, axis=0, keepdims=True
# )

# # ---------------------------------------------------------
# # 2️⃣ SVD-based PCA (numerically stable)
# # ---------------------------------------------------------

# U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)

# # Eigenfaces are right singular vectors
# eigenfaces_svd = Vt.T   # shape (D, N)

# # # Normalize eigenfaces
# # eigenfaces_svd = eigenfaces_svd / np.linalg.norm(
# #     eigenfaces_svd, axis=0, keepdims=True
# # )

# # ---------------------------------------------------------
# # 3️⃣ Align number of eigenfaces for fair comparison
# # ---------------------------------------------------------

# K = min(eigenfaces_cov.shape[1], eigenfaces_svd.shape[1])

# eigenfaces_cov = eigenfaces_cov[:, :K]
# eigenfaces_svd = eigenfaces_svd[:, :K]

# # ---------------------------------------------------------
# # 4️⃣ Cosine similarity comparison (CORRECT way)
# # ---------------------------------------------------------

# print("\nCosine similarity between Covariance PCA and SVD PCA eigenfaces:\n")

# for i in range(min(10, K)):
#     cos_sim = np.abs(
#         np.dot(eigenfaces_cov[:, i], eigenfaces_svd[:, i])
#     )
#     print(f"Eigenface {i+1}: cosine similarity = {cos_sim:.6f}")

# # ---------------------------------------------------------
# # 5️⃣ Visual comparison (optional but very informative)
# # ---------------------------------------------------------

# # i = 0  # try 0, 1, 2 ...

# # plt.figure(figsize=(8, 4))

# # plt.subplot(1, 2, 1)
# # plt.imshow(eigenfaces_cov[:, i].reshape(image_shape), cmap="gray")
# # plt.title("Covariance PCA")
# # plt.axis("off")

# # plt.subplot(1, 2, 2)
# # plt.imshow(eigenfaces_svd[:, i].reshape(image_shape), cmap="gray")
# # plt.title("SVD PCA")
# # plt.axis("off")

# # plt.suptitle(f"Eigenface {i+1} comparison")
# # plt.show()

# # ---------------------------------------------------------
# # 6️⃣ Model testing on test sample data
# # ---------------------------------------------------------

# k = min(20, eigenfaces_svd.shape[1])
# eigenfaces_svd_k = eigenfaces_svd[:, :k]
# print("Eigenface norm (first 5): ",
#       np.linalg.norm(eigenfaces_svd_k[:,:5], axis=0))

# X_train_PCA = X_train_centered @ eigenfaces_svd_k

# # PCA space reality
# # Each face is represented as a vector:
# #         z (projected) = [α1, α2, α3, α4........αk]
# # This vector encodes:
# # Direction → identity structure
# # Magnitude → illumination, contrast, background energy

# # eucledian_distance
# #     Direction Difference
# #     Magnitude Difference
# # cosine_distance
# #     Ignores magnitude
# #     focuses only on relative contribution of eigenfaces

# X_test_centered = X_test - mean_face
# X_test_PCA = X_test_centered @ eigenfaces_svd_k

# from app.infer import(
#     eucledian_distance,
#     cosine_distance,
#     predict_nearest_neighbor_single, 
#     predict_nearest_neighbor_multi,
#     infer_single_image
# )

# print("Enter the distance function to predict faces: 'euclidean' or 'cosine' ")
# Distance_function = input()

# # single image testing using X_test dataset
# y_pred_single = predict_nearest_neighbor_single(
#     X_test_PCA[0],
#     X_train_PCA,
#     y_train,
#     distance = Distance_function
# )

# print("Single True labels:", y_test[0])
# print("Single Predicted labels:", y_pred_single)

# # multi image testing using X_test dataset
# y_pred = predict_nearest_neighbor_multi(
#     X_test_PCA,
#     X_train_PCA,
#     y_train,
#     distance = Distance_function
# )

# accuracy = np.mean(y_pred == y_test)
# print(f"Test accuracy: {accuracy * 100:.2f}%")

# from collections import Counter

# print("True labels:", Counter(y_test))
# print("Predicted labels:", Counter(y_pred))

# # -------------------------------------------------------------------
# # 7️⃣ Accuracy vs K-Plot with Cosine vs Euclidean distance comparison
# # -------------------------------------------------------------------

# k_values = list(range(2, min(20, X_train_centered.shape[0])))
# accuracies_euclid= []
# accuracies_cosine= []

# for k in k_values:
#     eigenfaces_k = eigenfaces_svd[:, :k]

#     ## Ensure Contiguous (macOS safety)
#     eigenface_k = np.ascontiguousarray(eigenfaces_k)

#     X_train_PCAected = X_train_centered @ eigenface_k
#     X_test_PCAected = X_test_centered @ eigenface_k

#     y_pred_eucllid = predict_nearest_neighbor_multi(
#         X_test_PCAected,
#         X_train_PCAected,
#         y_train,
#         distance = "euclidean"
#     )
#     y_pred_cosine = predict_nearest_neighbor_multi(
#         X_test_PCAected,
#         X_train_PCAected,
#         y_train,
#         distance = "cosine"
#     )

#     acc_euclid = np.mean(y_pred_eucllid == y_test)
#     acc_cosine = np.mean(y_pred_cosine == y_test)

#     accuracies_euclid.append(acc_euclid)
#     accuracies_cosine.append(acc_cosine)

#     # print(F"k = {k:2d} -> accuracy_euclid = {acc_euclid*100:.2f}%  &  accuracy_cosine = {acc_cosine*100:.2f}%")

# # plt.figure(figsize=(7, 4))
# # plt.plot(k_values, accuracies_euclid, accuracies_cosine, marker ="o")
# # plt.xlabel("Number of Eigenfaces (k)")
# # plt.ylabel("Accuracy")
# # plt.title("Accuracy vs Number of Eigenface")
# # plt.grid(True)
# # plt.show()

# # Cosine ≥ Euclidean, especially at higher k
# # Interpretation:
#     # PCA projection produces features where direction matters more than magnitude
#     # Cosine distance is scale-invariant
#     # Euclidean distance is affected by energy differences (lighting, contrast)
# # This Means: Cosine distance is better suited for Eigenfaces.

# # Why cosine usually works better in Eigenfaces (intuition)
#     # After PCA:
#         # Each face is represented as a combination of eigenfaces
#         # The direction of the coefficient vector encodes identity
#         # The length can change due to lighting, contrast, background
#     # Cosine distance:
#         # Compares angles
#         # Ignores magnitude
#         # Focuses on identity structure
# # That’s why it often wins.

# # -------------------------------------------------------
# # 8️⃣ Confusion Matrix - Who gets mistaken for whom
# # -------------------------------------------------------
# # Accuracy hides:
#     # 1. Which identities are confused 
#     # 2. systematic errors

# # step 8.1: Compute Confusion Matrix
# num_classes = len(np.unique(y_test))
# conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

# for true_label, pred_label in zip(y_test, y_pred):
#     conf_matrix[true_label, pred_label] += 1

# print("Confusion Matrix: ")
# print(conf_matrix)

# # Optional but useful Pretty Print
# import pandas as pd

# print(Counter(y_test))
# labels = [f"subject_{i}" for i in range(num_classes)]
# df_cm = pd.DataFrame(conf_matrix, index= labels, columns=labels)

# print(df_cm)
# # Interpretation:
# # Rows → true identity
# # Columns → predicted identity
# # Diagonal → correct predictions

# # Step 8.2: Find misclassified indices
# print(y_test)       # print all test label
# print(y_pred)       # Print all predicted label by model ran on test data

# misclassified_idx = np.where(y_test != y_pred)[0]
# print(misclassified_idx)
# print(len(misclassified_idx))

# # Step 8.3: Print details for each failure
# for idx in misclassified_idx:
#     print(
#         f"Test index {idx}: "
#         f"True: subject_{y_test[idx]}, "
#         f"Predicted: subject_{y_pred[idx]}"

#     )
# # This already tells you:
#     # Which identities are hard
#     # Whether failures cluster around certain subjects

# # Step 8.3: Visualize misclassified Faces
# import matplotlib.pyplot as plt

# # Simple plot of actual misclassified face
# # for idx in misclassified_idx:
# #     img = X_test[idx].reshape(100, 100)  # original flattened image
# #     plt.imshow(img, cmap="gray")
# #     plt.title(
# #         f"True: subject_{y_test[idx]} | "
# #         f"Pred: subject_{y_pred[idx]}"
# #     )
# #     plt.axis("off")
# #     plt.show()

# # Print the images of testfaces generated based on training data
# face_name = {0: "Kush", 1: "Shrikant", 2: "Vanshita", 3: "sarvesh", 4: "Vishal"}

# # Detailed plot of actual misclassified face (actual, PCA reconstructed, Nearest Training face)
# for idx in misclassified_idx:
#     # Original face
#     original = X_test[idx].reshape(100,100)
    
#     # Reconstructed face using eigenfaces generated by training data
#     reconstructed_img = mean_face + X_test_PCA[idx] @ eigenfaces_svd_k.T
#     reconstructed_img = reconstructed_img.reshape(100,100)

#     # Nearest training face (predicted Identity)
#     if Distance_function == "euclidean":
#         distances = [eucledian_distance(X_test_PCA[idx],x_train) for x_train in X_train_PCA]
#     else:
#         distances = [cosine_distance(X_test_PCA[idx],x_train) for x_train in X_train_PCA]

#     nearest_idx = np.argmin(distances)

#     nearest_train_face = X_train[nearest_idx].reshape(100,100)

#     plt.figure(figsize=(10,4))

#     plt.subplot(1, 3, 1)
#     plt.imshow(original, cmap="gray")
#     plt.title(f"True Face: {face_name[y_test[idx]]}")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.imshow(reconstructed_img, cmap="gray")
#     plt.title(f"Reconstructed Face: {face_name[y_test[idx]]}")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.imshow(nearest_train_face, cmap="gray")
#     plt.title(f"Nearest Predicted: {face_name[y_pred[idx]]}")
#     plt.axis("off")

#     print(y_train[nearest_idx])

# plt.suptitle(f"True Face vs Reconstructed Face vs Neareset Predicted Face")
# plt.show()

# # plt.figure(figsize=(10, 3))
# # for i in range(len(y_test)):
# #     plt.subplot(2,7,i+1)
# #     reconstructed = mean_face + X_test_PCA[i] @ eigenfaces_svd_k.T        # shape(10000, )
# #     plt.imshow(X_test[i].reshape(100,100), cmap = "gray")
# #     plt.title(face_name[y_test[i]])
# #     plt.axis("off")
# # plt.suptitle("Test Faces") 
# # plt.show()


# # -------------------------------------------------------
# # 9️⃣ Infer Pipeline on new Image
# # -------------------------------------------------------

# processed_dir = Path("data/processed")

# for img_path in sorted(processed_dir.glob("*.png")):
#     pred_euc, pred_cos = infer_single_image(
#         img_path,
#         mean_face,
#         eigenfaces_svd_k,
#         X_train_PCA,
#         y_train
#     )

#     print(
#         f"{img_path.name:20s} → "
#         f"Euclidean: subject_{pred_euc}, "
#         f"Cosine: subject_{pred_cos}"
#     )

# # How to interpret results (important)
# # Case A — both metrics agree
# #     ✔ Very confident prediction
# # Case B — Euclid ≠ Cosine
# #     ✔ Ambiguous image
# #     ✔ Likely lighting / pose / background issue
# #     ✔ Exactly the behavior you analyzed earlier
