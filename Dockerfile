# syntax=docker/dockerfile:1
FROM --platform=linux/arm64 debian:12

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    software-properties-common \
    python3-full \
    python3-pip \
    jq \
    moreutils \
    git \
    tzdata && \
    rm -rf /var/lib/apt/lists/*


# Install DotNet SDK 10.0 per Microsoft's instructions. .NET 10 is the only
# version that Microsoft's Debian 12 feed publishes for arm64.
WORKDIR /opt
RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-10.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Dafny as a .NET tool (no Linux arm64 zip exists). Dafny 4.9.0 is built
# for .NET 6, so --allow-roll-forward lets it run on .NET 10. It has no bundled
# Z3, so it uses the z3 on the PATH (below).
WORKDIR /opt
RUN dotnet tool install dafny --version 4.9.0 --tool-path /opt/dafny --allow-roll-forward

ENV PATH="/opt/dafny:$PATH"

# Download and extract Z3 (arm64 build)
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.13.4/z3-4.13.4-arm64-glibc-2.34.zip && \
    unzip z3-4.13.4-arm64-glibc-2.34.zip && \
    rm z3-4.13.4-arm64-glibc-2.34.zip

ENV PATH="/opt/z3-4.13.4-arm64-glibc-2.34/bin:$PATH"

# Final working directory
WORKDIR /workspace

# Install tensorflow
RUN python3 -m venv cav2025-artifact && \
    ./cav2025-artifact/bin/pip install tensorflow==2.13.0

# The same packages as the native setup in README.md. Naming tensorflow==2.13.0
# again makes pip respect its limits. numpy 1.24.3 is the newest numpy that
# TensorFlow 2.13 accepts, and tensorflow-metadata 1.17.0 requires protobuf
# 4.25, the newest protobuf that it accepts: newer tensorflow-metadata releases
# need protobuf 6, and tensorflow_datasets then fails to import. ART 1.19.1 was
# current when the CAV 2025 artifact was made; ART 1.20.0 moved to Keras 3, and
# run_pgd_attack.py then fails.
RUN ./cav2025-artifact/bin/pip install tensorflow==2.13.0 numpy==1.24.3 tensorflow-metadata==1.17.0 tensorflow_datasets matplotlib scikit-learn pandas mpmath adversarial-robustness-toolbox==1.19.1

# Install vulnerable gloro
RUN git clone https://github.com/klasleino/gloro && \
    cd gloro && \
    git checkout a218dcdaaa41951411b0d520581e96e7401967d7 && \
    ../cav2025-artifact/bin/python setup.py build && \
    ../cav2025-artifact/bin/python setup.py install

# Copy over artifact files
RUN mkdir robustness-verifier
COPY . robustness-verifier
COPY cav2025-models.zip .
RUN unzip cav2025-models.zip
