# Inference container for CHIMERA 2025 Task 3 (Team NMIL).
# Base: PyTorch 1.13.1 / CUDA 11.6 (Grand-Challenge compatible).
FROM --platform=linux/amd64 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS example-algorithm-amd64

ENV PYTHONUNBUFFERED=1

# System libraries for OpenSlide / libvips / OpenCV.
# The ubuntu-toolchain-r PPA is added to obtain a newer libvips than the base image provides.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libvips-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    libopenslide-dev \
    pkg-config \
    libtiff5-dev \
    libjpeg-turbo8-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Non-root user, as in the Grand-Challenge algorithm template.
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# Model artefacts are NOT part of this repository; see README ("Model weights").
# Expected layout before building:
#   model/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin
#   model/bcg_best_model_epoch_9.pth
#   model/bcg_preprocessor.pkl
COPY --chown=user:user model /opt/app/model/

COPY --chown=user:user inference.py feature_extractor.py my_survival_model.py preprocessor.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
