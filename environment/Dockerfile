################################################################################
# Multi-stage Dockerfile (builder + runtime)
# - builder: installs build tools, creates a Python virtualenv and installs all
#   pip packages (including the packaged TensorFlow wheel). This layer is
#   cacheable for faster iterative builds when dependencies change.
# - runtime: lightweight image that contains only runtime OS libs and the
#   prebuilt virtualenv copied from the builder stage. This keeps the final
#   image smaller and rebuilds faster when only application files change.
################################################################################

FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Use faster mirrors
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.edge.kernel.org/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.edge.kernel.org/ubuntu/|g' /etc/apt/sources.list

# Build-time packages needed for wheels / compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libcairo2-dev libfribidi-dev pkg-config python3-dev wget \
    libglib2.0-0 libpango1.0-dev libpangocairo-1.0-0 libharfbuzz-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv to copy into runtime stage
ENV VENV_DIR=/opt/venv
RUN python3 -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# Copy local TensorFlow wheel into the builder and install packages into venv
COPY tensorflow-2.8.0-cp38-cp38-manylinux2010_x86_64.whl /tmp/
# Copy any pre-downloaded wheels (e.g., torch, torchvision) to avoid large pip downloads during build
# Place wheels in environment/wheels/ on the host before building the image.
COPY wheels /tmp/wheels
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    jupyter==1.1.1 \
    matplotlib==3.7.5 \
    mplcairo==0.6.1 \
    numpy==1.24.4 \
    opencv-python==4.12.0.88 \
    pangocairocffi==0.7.0 \
    pillow==10.4.0 \
    protobuf==3.20.0 \
    pycairo==1.26.1 \
    scikit-learn==1.3.2 \
    tqdm==4.67.1

# Install TensorFlow wheel separately so it can be cached independently
RUN pip install --no-cache-dir /tmp/tensorflow-2.8.0-cp38-cp38-manylinux2010_x86_64.whl

# Cleanup builder artefacts
RUN rm -f /tmp/tensorflow-2.8.0-cp38-cp38-manylinux2010_x86_64.whl

# Install PyTorch and torchvision via conda (prebuilt, avoids large pip downloads)
# If wheel files for torch/torchvision were copied into /tmp/wheels, install them first.
# This avoids streaming very large files from PyPI and makes builds reproducible.
RUN if [ -d /tmp/wheels ] && ls /tmp/wheels/*.whl 1> /dev/null 2>&1; then \
            echo "Installing local wheel files from /tmp/wheels"; \
            pip install --no-cache-dir /tmp/wheels/*.whl || true; \
        else \
            echo "No local wheels found; falling back to conda install for PyTorch"; \
            conda install -y -c pytorch pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7 || true; \
        fi


################################################################################
# Runtime image: smaller image with only runtime OS packages + the venv
################################################################################
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.edge.kernel.org/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.edge.kernel.org/ubuntu/|g' /etc/apt/sources.list

# Runtime OS packages (fonts, GL, cairo/pango etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip fontconfig fonts-dejavu-core fonts-freefont-ttf libgl1 libglib2.0-0 uuid-runtime \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt virtualenv from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy any project-level scripts and run post-install (keeps scripts in final image)
COPY postInstall /postInstall
RUN chmod +x /postInstall && /postInstall