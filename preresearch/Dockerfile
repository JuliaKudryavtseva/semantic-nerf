# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation
# Built on Thu_Nov_18_09:45:30_PST_2021
# Cuda compilation tools, release 11.5, V11.5.119
# Build cuda_11.5.r11.5/compiler.30672275_0
# Arguments to build Docker Image using CUDA

FROM ubuntu:22.04

# python
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# torch
RUN pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

# stuff
RUN pip install numpy tqdm opencv-python matplotlib
RUN apt update && apt install unzip zip 
# RUN apt update && apt install wget
# git 
RUN apt update && apt install -y git

# clone sam with text and install dependecies in CPU model
RUN git clone https://github.com/luca-medeiros/lang-segment-anything.git

WORKDIR /lang-segment-anything
RUN pip install -e .

# folder for video to segment
RUN mkdir dataset 
RUN mkdir vis_prompt

# copy files to execute 
COPY preresearch/segment_frames.py segment_frames.py
COPY preresearch/vis_seg.py vis_seg.py