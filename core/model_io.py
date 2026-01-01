import numpy as np
from pathlib import Path

def save_model(
    path: Path,
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    lda_faces: np.ndarray | None = None
):
    """
    Save trained PCA/LDA model components.
    """
    path.mkdir(parents=True, exist_ok=True)

    np.savez(
        path / "model.npz",
        mean_face=mean_face,
        eigenfaces=eigenfaces,
        lda_faces=lda_faces
    )

def load_model(path: Path):
    """
    Load trained PCA/LDA model components.
    """
    data = np.load(path / "model.npz", allow_pickle=True)

    return (
        data["mean_face"],
        data["eigenfaces"],
        data["lda_faces"] if "lda_faces" in data.files else None
    )
