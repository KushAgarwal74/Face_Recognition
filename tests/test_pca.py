import numpy as np

from core.pca import fit_pca, project, reconstruct
from core.lda import fit_lda


def test_fit_pca_shapes():
    """
    PCA should return correctly shaped outputs.
    """
    N, D, k = 10, 20, 5
    X = np.random.rand(N, D)

    mean_face, eigenfaces, eigenfaces_k = fit_pca(X, k)

    assert mean_face.shape == (D,)
    assert eigenfaces.shape == (D, min(N, D))
    assert eigenfaces_k.shape == (D, k)


def test_pca_projection_shape():
    """
    Projected data should have shape (N, k).
    """
    N, D, k = 8, 16, 4
    X = np.random.rand(N, D)

    mean_face, _, eigenfaces_k = fit_pca(X, k)
    X_proj = project(X, mean_face, eigenfaces_k)

    assert X_proj.shape == (N, k)


def test_pca_reconstruction_shape():
    """
    Reconstructed data should have original shape (N, D).
    """
    N, D, k = 6, 12, 3
    X = np.random.rand(N, D)

    mean_face, _, eigenfaces_k = fit_pca(X, k)
    X_proj = project(X, mean_face, eigenfaces_k)
    X_recon = reconstruct(X_proj, mean_face, eigenfaces_k)

    assert X_recon.shape == (N, D)

def test_lda_output_dim():
    import numpy as np
    from core.lda import fit_lda

    X = np.random.rand(10, 5)
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    W = fit_lda(X, y, num_components=4)

    assert W.shape == (5, 4)
