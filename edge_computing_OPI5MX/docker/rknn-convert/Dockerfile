FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential cmake \
    libglib2.0-0 libgl1-mesa-glx libgl1 libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "numpy<=1.26.4" \
    "onnx==1.16.1" \
    "onnxoptimizer==0.3.8" \
    "onnxruntime>=1.16.0" \
    "protobuf==3.20.3" \
    "opencv-python-headless>=4.5.5.64" \
    "scipy>=1.9.3" \
    "tqdm>=4.64.1" \
    "fast-histogram>=0.11" \
    "psutil>=5.9.0" \
    "ruamel.yaml>=0.17.21" \
    "Pillow>=10.0.1"

RUN pip install --no-cache-dir rknn-toolkit2==2.3.2

WORKDIR /workspace
CMD ["bash"]
