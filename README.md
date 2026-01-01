![CI](https://github.com/KushAgarwal74/Face_Recognition/actions/workflows/github-actions.yml/badge.svg)

# Face Recognition using PCA & LDA (Eigenfaces & Fisherfaces)

This project implements a **classical face recognition pipeline** using:
- PCA (Eigenfaces) for dimensionality reduction
- LDA (Fisherfaces) for class separability
- Nearest Neighbor classifier
- Fully reproducible with Python, Docker, CI, and Kubernetes

---

## 📂 Project Structure

face_recognition/
├── app/ # Training, inference, evaluation
├── core/ # PCA, LDA, metrics (math layer)
├── data/ # Sample dataset (small, included)
├── configs/ # YAML configuration
├── docker/ # Dockerfile
├── k8s/ # Kubernetes job
├── CI/ # GitHub Actions CI
├── requirements.txt
└── README.md

All parameters are configurable via:
  configs/config.yaml


## 🚀 Quick Start (Local)

### 1️⃣ Clone repository
```bash
git clone https://github.com/KushAgarwal74/<REPO_NAME>.git
cd <REPO_NAME>
```

2️⃣ Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

3️⃣ Install dependencies
```
pip install -r requirements.txt
```

3️⃣ Install dependencies
```
python -m app.train
```
Expected output:
```
Feature space   : PCA + LDA
Test accuracy  : ~93%
Confusion matrix:
```

🐳 Run with Docker
```
docker build -t face-recognition -f docker/Dockerfile .
docker run --rm face-recognition
```
☸ Run with Kubernetes (local cluster)
```
kubectl apply -f k8s/job-train.yaml
kubectl logs job/face-recognition-train
```

📌 Notes
Dataset is intentionally small (36 images) for reproducibility.
Pipeline is extensible to larger datasets and CNN-based models.


## 🛠 Makefile Commands

```bash
make venv        # Create virtual environment
make install     # Install dependencies
make train       # Run training
make docker      # Build Docker image
make clean       # Remove artifacts

