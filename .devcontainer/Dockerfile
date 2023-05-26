FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libsndfile-dev

RUN pip install \
    pylint \
    autopep8 \
    tqdm \
    matplotlib \
    seaborn \
    scipy \
    pytest \
    pandas \
    catboost \
    scikit-learn \
    nspfile \
    librosa \
    tensorboard \
    torch-tb-profiler \
    xlrd \
    openpyxl \
    wfdb \
    beautifulsoup4 \
    dask \
    praat-parselmouth