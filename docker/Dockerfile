# Use the specified TensorFlow GPU image with version 2.13.0
FROM tensorflow/tensorflow:2.13.0-gpu

# Install dependencies for building Python from source
RUN apt-get update && \
    apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    curl \
    libbz2-dev && \
    rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11.8 from source
RUN wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz && \
    tar -xf Python-3.11.8.tgz && \
    cd Python-3.11.8 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.8 Python-3.11.8.tgz

# Set Python 3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Reinstall TensorFlow to ensure compatibility with Python 3.11
RUN pip install tensorflow==2.13.0

# Install additional Python packages
RUN pip install tensorflow-datasets pillow

# Set environment variables
ENV noninteractive=true
ENV HOME=/workspace

# Set the working directory
WORKDIR /workspace
