FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

RUN pip install numpy torch torchvision tqdm opencv-python matplotlib
RUN apt update && apt install -y wget

WORKDIR /workspace
