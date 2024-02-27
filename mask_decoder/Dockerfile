FROM ubuntu:22.04

# install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

RUN pip install numpy torch torchvision tqdm opencv-python matplotlib


# wget
RUN apt update && apt install -y wget
# git 
RUN apt update && apt install -y git


RUN mkdir weights && cd weights && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  && cd ..

WORKDIR /workspace
