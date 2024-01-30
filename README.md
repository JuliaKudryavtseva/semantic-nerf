# Semantic-nerf

## Preresearch

```
# create docker image and start docker container
docker build -t sem_nerf.preresearch -f preresearch/Dockerfile .

docker run -it --rm \
           -v $PWD/preresearch/assets:/lang-segment-anything/dataset \
           -v $PWD/preresearch/vis_prompt:/lang-segment-anything/vis_prompt \
           --name kudryavtseva.preresearch \
           kudryavtseva.preresearch


# extract frames from video
cd dataset && python3 video2images.py && cd ..


# genereate masks for text prompt 'brown teddy bear'
python3 segment_frames.py --text 'brown teddy bear' 

# visualise results of segmentation in folder "vis_prompt"
python3 vis_seg.py --exp-name 'brown teddy bear' --out-name 'brown_teddy_bear'

```

## Research

Create image

```
docker build --tag kudryavtseva/nerfstudio:version1 -f Dockerfile .
```

Docker container 

```
docker run --gpus all \                                                                               
             -v /folder/of/your/data:/workspace/ \               
             -v /home/j.kudryavtseva/.cache/:/home/user/.cache/ \   
             -u root \
             -p 7087:7007 \                                      
             --rm \                                              
             -it \                                               
             --memory=50gb \ 
             --shm-size=50gb \                                  
             kudryavtseva/nerfstudio:version1            
```

```
docker run --gpus all -v /home/j.kudryavtseva/.cache/:/home/user/.cache/ -u root -p 7087:7087 --rm -it --memory=50gb kudryavtseva/nerfstudio:version1   
```


cd method_nerf/
pip install -e .
ns-install-cli
cd ..

ns-train clip-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087