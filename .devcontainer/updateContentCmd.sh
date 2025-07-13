#!/bin/bash

# cp ./.devcontainer/sources.txt /etc/apt/sources.list

apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    curl \
    sox \
    openssh-server \
    ffmpeg \
    libgl1-mesa-glx \
    git nano wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*    

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py
    
pip install -r /workspaces/Kimi-Audio/requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation

