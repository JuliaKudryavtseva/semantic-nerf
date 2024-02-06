# Semantic-nerf

## Research

### Extract SAM features for images

```
mkdir data

data/ teatime/(images + json)
```

```
DATA_PATH=teatime

docker build -t kudryavtseva.sam_features -f image_segmentation/Dockerfile  .

docker run --rm --gpus device=5 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/seg_masks/segment-anything/dataset \
            -v $PWD/assets:/seg_masks/segment-anything/assets \
            --name kudryavtseva.sam_features \
            kudryavtseva.sam_features

```
### Semantic-NeRF
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


### Decode masks

```
DATA_PATH=teatime

docker build -t kudryavtseva.decoder -f mask_decoder/Dockerfile  .

docker run -it --rm \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/mask_decoder:/workspace \
            -v $PWD/data:/workspace/dataset \
            -v $PWD/assets:/workspace/assets \
            --name kudryavtseva.decoder \
            kudryavtseva.decoder

pip install -e .

python3 decode_masks.py
```