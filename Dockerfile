# Circular Fingerprint AutoML TPOT Explainability
# Syntax: docker build -t circular-fingerprint-tpot .
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# System deps: RDKit (Debian), libs for matplotlib/XGBoost, certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-rdkit rdkit-data \
      libxrender1 libxext6 libsm6 libglib2.0-0 \
      libgomp1 \
      git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Workdir and source
WORKDIR /workspace
COPY . /workspace

# Optimized for circular fingerprints and TPOT AutoML
ENV MPLBACKEND=Agg \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Python deps optimized for TPOT and circular fingerprints
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.23.5 pandas==2.0.* scikit-learn==1.1.* matplotlib==3.7.* pillow openpyxl \
      tpot==0.12.1 xgboost==1.7.* lime==0.2.* deepchem==2.7.1

# Default: Train TPOT model and generate interpretations
CMD ["python", "models/train_tpot_simple.py"]
