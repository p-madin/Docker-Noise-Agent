#!/bin/sh
cd /root/Workspace

# Dataset Provisioning
echo "Checking datasets..."

# ESC-50 Dataset
if [ ! -d "datasets/ESC-50-master" ]; then
    echo "ESC-50 not found. Downloading..."
    mkdir -p datasets
    cd datasets
    curl -L -o ESC-50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip -q ESC-50.zip
    rm ESC-50.zip
    cd /root/Workspace
    echo "ESC-50 downloaded and extracted."
else
    echo "ESC-50 already exists."
fi

# UrbanSound8K Dataset
if [ ! -d "datasets/UrbanSound8K" ]; then
    echo "UrbanSound8K not found. Downloading..."
    mkdir -p datasets
    cd datasets
    curl -L -o UrbanSound8K.tar.gz "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
    tar -xzf UrbanSound8K.tar.gz
    rm UrbanSound8K.tar.gz
    cd /root/Workspace
    echo "UrbanSound8K downloaded and extracted."
else
    echo "UrbanSound8K already exists."
fi

echo "Dataset check complete."

# Start the application
python server.py