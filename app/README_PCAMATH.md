# Responsibilities
    # Receive X_train
    # Compute mean face
    # Center data
    # Compute PCA / LDA
    # Train model

# import numpy as np
# import matplotlib.pyplot as plt

# from app.dataset import (
    build_dataset,
    train_test_split_stratified
)

# Build Full Dataset
# X, y = build_dataset()

# Split into train/test dataset
# X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_ratio = 0.3)

mean_face = np.mean(X_train, axis = 0)

# mean_face_img = mean_face.reshape(100, 100)
# test is the mean face is blurry and have images visuals
# plt.imshow(mean_face_img, cmap="gray")
# plt.title("Mean Face (Training Data)")
# plt.axis("off")
# plt.show()

X_train_centered = X_train - mean_face
# This is vectorized subtraction:  
# NumPy subtracts mean_face from every row
# This removes:
# overall brightness$
# common facial structure
# background that is constant across images

# If data is not centered:
    # The mean dominates variance
    # First eigenvector points toward the mean
    # Eigenfaces are meaningless
# With centered data:
    # PCA captures differences between faces
    # Eigenfaces represent facial variations
    # Mathematically:
# PCA assumes data is centered at the origin.

# Each row of X_train_centered answers:
# “How is this face different from the average face?”
# Centering does not change relative geometry or  pixel orientation in space
# It moves the origin to the mean face
# Pixel axes still exist, but values now mean deviation, not brightness
# pixel values no longer represent brightness They represent deviations from average
# Examples:
# Positive values → brighter than average at that pixel
# Negative values → darker than average
# Zero → same as average

# PCA finds the best k-dimensional subspace
# Always passes through origin because data is centered


# Step 2:
# PCA needs the covariance matrix, direct covariance is 10000x10000 
# here the trick is computes it in sample sapace of 22 x 22

# Step2.1. Covariance Matrix (Eigenfaces trick)
N, D = X_train_centered.shape   
C = X_train_centered @ X_train_centered.T / (N - 1)  # (22x10000)x(10000x22)
# similarity between faces in centered space
# “How similar image i and image j are in how they deviate from the mean face”
# This is face-to-face similarity, not pixel-to-pixel.
    # Large positive → images deviate from mean in the same direction
    # Negative → deviate in opposite directions
    # Zero → unrelated deviations


# Step2.2. Eigen decomposition (this creates PCA directions)
# Why this step exists
# Eigenvectors here are PCA directions in image space with linear combination of faces
# Eigenvalues quantify variance along those direction

eigenvalues, eigenvectors = np.linalg.eigh(C)

# Because C is:
# symmetric
# real-valued
# We use np.linalg.eigh (important)
# eigh() returns values in ascending order
# PCA needs largest variance first
# therefore we need to sort in descending order

print("Eigenvalues shape:", eigenvalues.shape)
print("Eigenvectors shape:", eigenvectors.shape)


# Step 2.3: Sort eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# STEP 4.4: Compute eigenfaces (pixel space)
# Right now:
# Eigenvectors live in sample space (22D)
# Eigenfaces must live in pixel space (10000D), which is basically eigenvector in pixel space
# This mapping is the core Eigenfaces equation where,
# You convert image-space PCA directions into pixel-space directions

eigenfaces = X_train_centered.T @ eigenvectors      # each column is eigenface

# print("Eigenvalues:", eigenvalues[:10])
# print("Min eigenvalue:", np.min(eigenvalues))
# print("Any NaN in eigenfaces before norm?", np.isnan(eigenfaces).any())


# STEP 4.5: Normalize eigenfaces
# Why normalization matters
    # Before normalization:
        # “Some directions are longer just because of scaling”
    # After normalization:
        # “All directions are unit arrows; only direction matters”
print(eigenfaces)
eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

# This is the core PCA extraction.
    # Eigenfaces now represent pure directions of variation
    # Projection values become meaningful coefficients
    # Distance comparisons become valid

# Step 3
# We’ll proceed in this exact order:
    # Choose k (dimensionality)
    # Project training data
    # Reconstruct a face
    # Sanity checks (this tells you PCA truly worked)

# step 6.2 Select top k eigenfaces
k = 8
eigenfaces_k = eigenfaces[:, :k]            # shape (10000, k)
# each column is one eigen face

# Step 6.3 Project training data
# This is where dimensionality reduction actually happens.

X_train_proj = X_train_centered @ eigenfaces_k
# shape (22, k) means each face -> now represents k numbers instead of 10,000 pixels
# these k numbers are eigenface coefficients
# this interprets “Where does this face lie along each eigenface direction?”
# [α₁, α₂, ..., αₖ]
# “How much does this face vary along eigenface i?
# 🧠 Intuition (key)
# Eigenfaces = axes
# Projection = coordinates in this new coordinate system

# Step 6.4 Visualize eigenfaces
# plt.figure(figsize=(10, 4))
# for i in range(min(k, 10)):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(eigenfaces_k[:, i].reshape(100, 100), cmap="gray")
#     plt.title(f"Eigenface {i+1}")
#     plt.axis("off")

# plt.suptitle("Top Eigenfaces")
# plt.show()

# Step 6.5 Reconstruct a face (ultimate test)
# i = 7
# reconstructed = mean_face + X_train_proj[i] @ eigenfaces_k.T        # shape(10000, )
# plt.imshow(reconstructed.reshape(100, 100), cmap="gray")
# plt.title("Reconstructed Face")
# plt.axis("off")
# plt.show()

# What reconstruction tells you
# | Outcome                  | Interpretation |
# | ------------------------ | -------------- |
# | Face recognizable        | PCA correct    |
# | Slight blur              | Expected       |
# | More blur with smaller k | Normal         |
# | Noise or garbage         | Bug earlier    |




