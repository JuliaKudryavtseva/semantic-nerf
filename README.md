# Semantic-nerf

## Research

### Extract SAM features for images

https://huggingface.co/datasets/ethanweber/Mip-NeRF_360_Processed_with_Nerfstudio/tree/main

```
mkdir data

data/ teatime/(images + json)

 pip install nerfbaselines
 nerfbaselines download-dataset mipnerf360/kitchen -o kitchen
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

docker run -it --rm --gpus device=5  --memory=100gb --shm-size=100gb -e "DATA_PATH=$DATA_PATH" -v $PWD:/workspace -v /home/j.kudryavtseva/.cache/:/home/user/.cache --name kudryavtseva.sem_nerf -u root gbobrovskikh.nerfstudio:dev   


pip install -e . 
ns-install-cli

# train NeRF

ns-train sam-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087


# render sam-features # render rgb

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split test

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split val

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names rgb --split test
ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names rgb --split val





# view pre-trained
ns-viewer --load-config outputs/teatime/sam-nerf/2024-02-21_120120/config.yml --viewer.websocket-port=7087


```


# Train compression block
```


docker build -t kudryavtseva.compression_block -f compression_block/Dockerfile  .

docker run -it --rm --gpus device=3 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/compression_block:/workspace \
            -v $PWD/data:/workspace/data \
            -v $PWD/assets:/workspace/assets \
            -v $PWD/renders:/workspace/renders \
            --name kudryavtseva.compression_block \
            kudryavtseva.compression_block

python3 training.py
python3 evaluation.py
```


### Decode masks

```

DATA_PATH=teatime

docker build -t kudryavtseva.decoder -f mask_decoder/Dockerfile  .

docker run -it --rm \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/mask_decoder:/workspace \
            -v $PWD/data:/workspace/data \
            -v $PWD/assets:/workspace/assets \
            -v $PWD/renders:/workspace/renders \
            --name kudryavtseva.decoder \
            kudryavtseva.decoder

pip install -e .

./test.sh $DATA_PATH
```


# checkpoints

## teatime:

with reg: 2024-04-02_224507
no reg: 2024-04-04_144538

# espresso 

no reg: 2024-04-05_125843
reg: 2024-04-05_220246

# waldo_espress
reg: 2024-04-06_205615
no reg: 2024-04-07_123748

# dozer_nerfgun_waldo

reg: 2024-04-07_164143