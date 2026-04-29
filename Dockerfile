# maniskill
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opendilab

RUN apt-get update && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
    gcc g++ swig git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    libegl1 libgles2 libgl1 libglvnd0 libglx0 \
    libosmesa6 libxrender1 libxext6 libsm6 \
    mesa-utils libopengl0 \
    libgl1-mesa-dev libglu1-mesa libglu1-mesa-dev \
    libosmesa6-dev freeglut3-dev \
    libvulkan1 vulkan-tools \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN python -m pip install --upgrade pip setuptools wheel