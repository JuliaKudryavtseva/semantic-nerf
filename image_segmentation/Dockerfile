FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

RUN pip install numpy torch torchvision tqdm opencv-python


# wget
RUN apt update && apt install -y wget
# git 
RUN apt update && apt install -y git


RUN mkdir seg_masks 
RUN mkdir dataset

RUN cd seg_masks && git clone https://github.com/facebookresearch/segment-anything.git

WORKDIR /seg_masks/segment-anything/
RUN mkdir weights && cd weights && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  && cd ..


COPY image_segmentation/gt_sam_features.py  gt_sam_features.py


CMD python3 gt_sam_features.py --data-path $DATA_PATH --exp-name $DATA_PATH
