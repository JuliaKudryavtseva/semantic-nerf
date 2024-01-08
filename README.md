# Semantic-nerf

## Preresearch

```
cd preresearch

docker build -t kudryavtseva.preresearch .
docker run -it --rm -v $PWD:/preresearch --name kudryavtseva.preresearch kudryavtseva.preresearch

# clone lang-sam and install dependecies
git clone https://github.com/luca-medeiros/lang-segment-anything.git
cd lang-segment-anything 
pip install -e .

# move files to lang-segment-anything folder
cd .. && cp -r assets lang-segment-anything/dataset 
cp segment_frames.py lang-segment-anything/segment_frames.py && cp vis_seg.py lang-segment-anything/vis_seg.py

# extract frames from video
cd lang-segment-anything/dataset && python3 video2images.py && cd ..

# genereate masks for text prompt "bear"
python3 segment_frames.py --text bear 

# visualise results of segmentation in folder "vis_prompt"
python3 vis_seg.py --exp-name bear --out-name bear

```
