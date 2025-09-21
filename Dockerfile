# Syntax: docker build -t xaisar .
# Fast pip-only image with RDKit from apt (no conda needed)
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

# Silence noisy libs and use non-GUI backend
ENV MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_VERBOSITY=error \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Python deps (CPU only) - pinned for compatibility with TF 2.12 and DeepChem 2.7.1
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.23.5 pandas==2.0.* scikit-learn==1.1.* matplotlib==3.7.* pillow openpyxl \
      xgboost==1.7.* deepchem==2.7.1 tensorflow==2.12.* keras==2.12.* \
      transformers==4.41.* tokenizers==0.19.* tpot==0.12.1 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.*

# Default: generate all GIFs
CMD ["python", "visualization/generate_gifs.py"]
