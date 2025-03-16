FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libsndfile-dev \
    unzip \
    wget

RUN pip install pipenv