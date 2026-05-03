# Circular Fingerprint AutoML TPOT Explainability + QSAR Analysis + GPU Support
# Syntax: docker build -t explainable-ai-sar .
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3-dev \
      python3-rdkit rdkit-data \
      libxrender1 libxext6 libsm6 libglib2.0-0 \
      libgomp1 libopenblas-dev liblapack-dev gfortran \
      graphviz git ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Workdir and source
WORKDIR /workspace
COPY . /workspace

# Optimized for circular fingerprints and TPOT AutoML
ENV MPLBACKEND=Agg \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Python deps optimized for TPOT, circular fingerprints, QSAR analysis, and GPU acceleration
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.23.5 pandas==2.0.* scikit-learn==1.1.* matplotlib==3.7.* pillow openpyxl \
      seaborn==0.12.* joblib==1.2.* \
      tpot==0.12.1 xgboost==1.7.* lime==0.2.* deepchem==2.7.1 \
      shap==0.46.* \
      umap-learn==0.5.* hdbscan==0.8.* \
      rdkit-pypi==2022.9.5 \
      scipy>=1.9.0 statsmodels>=0.13.0 \
      plotly>=5.0.0 kaleido>=0.2.1 \
      networkx>=2.6 python-Levenshtein>=0.20.0 \
      cupy-cuda12x numba llvmlite && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu122

# Default: Train TPOT model and generate interpretations
CMD ["python", "models/train_tpot_simple.py"]
