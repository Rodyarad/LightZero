# Start from Ubuntu 20.04
FROM ubuntu:20.04

# Set the working directory in the Docker image
WORKDIR /opendilab

# Install Python 3.8 and other dependencies
# We update the apt package list, install Python 3.8, pip, compilers and other necessary tools.
# After installing, we clean up the apt cache and remove unnecessary lists to save space.
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip gcc g++ swig git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for Python and pip
# This makes it easy to call python and pip from any location in the container.
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Update pip and setuptools to the latest version
# This step ensures that we have the latest tools for installing Python packages.
RUN python -m pip install --upgrade pip setuptools
