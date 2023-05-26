FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libsndfile-dev \
    unzip \
    wget

RUN pip install \
    aiofiles==23.1.0 \
    aiohttp==3.8.4 \
    beautifulsoup4==4.11.1 \
    black==22.10.0 \
    catboost==1.2 \
    dask==2023.5.0 \
    flake8==6.0.0 \
    librosa==0.10.0.post2 \
    matplotlib==3.7.1 \
    nspfile==0.1.4 \
    openpyxl==3.1.2 \
    pandas==2.0.1 \
    praat-parselmouth==0.4.3 \
    pylint==2.17.4 \
    pytest==7.3.1 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    seaborn==0.12.2 \
    tensorboard==2.13.0 \
    torch-tb-profiler==0.4.1 \
    tqdm==4.64.1 \
    wfdb==4.1.1 \
    xlrd==2.0.1