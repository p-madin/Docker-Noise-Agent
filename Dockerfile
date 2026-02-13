FROM nvidia/cuda:12.3.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install Python and essential system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    pkg-config \
    wget \
    libsndfile1 \
    ffmpeg \
    curl \
    unzip \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp/requirements.txt
# Use --ignore-installed to bypass system package conflicts
RUN pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

COPY ./startup.sh /usr/local/startup.sh
RUN chmod +x /usr/local/startup.sh

WORKDIR /root/Workspace
CMD ["/usr/local/startup.sh"]