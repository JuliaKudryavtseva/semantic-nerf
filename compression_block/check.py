import numpy as np 
import os
import matplotlib.pyplot as plt


path = 'data/teatime/segmentation_results'
full_path = 'data/teatime/segmentation_results/rest_features'

if __name__ == '__main__':

    mean_rest_ft = np.load(os.path.join(path, 'rest_features.npy'))
    print(f'{mean_rest_ft.shape=}')

    frames_ft = os.listdir(full_path)

    for ind, frame in enumerate(frames_ft, 1):
        path_frame = os.path.join(full_path, frame)
        frame_rest_ft = np.load(path_frame)

        metrics = (frame_rest_ft - mean_rest_ft)**2
        plt.imshow(metrics[0, 4, ...])
        plt.show()
        plt.savefig(f'vis/{ind}.jpg')
        plt.clf()
