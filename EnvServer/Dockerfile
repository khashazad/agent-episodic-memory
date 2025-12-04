# Base image with Java 8 installed (Eclipse Temurin)
FROM eclipse-temurin:8-jdk

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        libxtst6 \
        libxi6 \
        libxrender1 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxss1 \
        libxxf86vm1 \
        xvfb \
        x11-xserver-utils \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

# Put conda on PATH
ENV PATH="/opt/miniconda/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# 1) Accept Anaconda TOS + create env and install Python + basic tooling
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n minerl python=3.9 -y && \
    conda run -n minerl pip install "pip<24.1" setuptools wheel

# 2) Pre-install gym 0.19.0 via conda (binary package)
RUN conda install -n minerl -c conda-forge "gym==0.19.0" -y

# 3) Download, patch, and install MineRL 0.4.4
RUN conda run -n minerl pip download --no-deps minerl==0.4.4 -d /tmp && \
    cd /tmp && \
    tar xf minerl-0.4.4.tar.gz && \
    cd minerl-0.4.4 && \
    sed -i "s/com.github.SpongePowered:MixinGradle:dcfaf61/org.spongepowered:mixingradle:0.6-SNAPSHOT/" \
        minerl/Malmo/Minecraft/build.gradle && \
    sed -i "/mavenCentral()/a \        maven { url 'https://repo.spongepowered.org/repository/maven-public/' }" \
        minerl/Malmo/Minecraft/build.gradle && \
    conda run -n minerl pip install .

# 4) Install the rest of your Python deps (WITHOUT minerl in requirements.txt)
RUN conda run -n minerl pip install -r requirements.txt

# Copy your Flask/MineRL server
COPY env_server.py .

# Expose Flask port
EXPOSE 5000

# Run the server in the minerl env
ENV DISPLAY=:99
CMD ["bash", "-lc", "Xvfb :99 -screen 0 1920x1080x24 & sleep 3; conda run --no-capture-output -n minerl python env_server.py"]