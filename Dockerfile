FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git unzip ca-certificates \
        build-essential \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# ✅ 接受 ToS（关键）
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n orenet python=3.8 -y \
    && conda clean -afy

SHELL ["conda", "run", "-n", "orenet", "/bin/bash", "-c"]

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
       --extra-index-url https://download.pytorch.org/whl/cu117

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "orenet"]
CMD ["/bin/bash"]
