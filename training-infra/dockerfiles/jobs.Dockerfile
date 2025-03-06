FROM ubuntu:20.04

# Install required packages in a single layer
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in a separate layer
RUN pip3 install --no-cache-dir huggingface_hub datasets

RUN  wget https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x mc \
    && mv mc /usr/local/bin/

# Default command when container starts
CMD ["/bin/bash"]
