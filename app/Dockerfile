FROM tensorflow/tensorflow:latest-gpu-jupyter
#FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && \
    apt-get install -y \
    g++ \
    git \
    wget \
    openjdk-8-jdk \
    python3-pip \
    curl 

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --ignore-installed

RUN git config --global http.postBuffer 524288000

# Set non-interactive and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul



