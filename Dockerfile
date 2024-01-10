# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set the working directory in the container
WORKDIR /workspace

# Copy preinstall.sh into the container at /workspace
COPY preinstall.sh /workspace/

# Check CUDA version
RUN /usr/local/cuda/bin/nvcc --version

# Run preinstall.sh for required dependencies
RUN sh /workspace/preinstall.sh

# Uninstall any existing Apex to ensure a clean installation
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :

# Install packaging before installing Apex
RUN pip install pip==23.1 && pip install packaging

# Install git before cloning Apex
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update \
    && apt-get install -y git

# Clone and install Apex
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
RUN python setup.py install --cuda_ext --cpp_ext

# Back to workspace
WORKDIR /workspace

# Copy all remaining directories and files into the container at /workspace
COPY . /workspace/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the command to run your application
CMD ["python", "app.py"]