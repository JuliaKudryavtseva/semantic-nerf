from torchmetrics import JaccardIndex
import numpy as np
import torch
from PIL import Image
import os

def calulate_ious(pred_masks: list, gt_masks: list):
    jaccard = JaccardIndex(task='binary')

    IOUs = []
    for ind in range(len(pred_masks)):

        target = torch.tensor(gt_masks[ind])
        pred = torch.tensor(pred_masks[ind])
        iou = jaccard(pred, target)
        iou = iou.item()

        if target.sum().item() == pred.sum().item()  == 0:
            iou=1
            
        IOUs.append(iou)
    return np.mean(IOUs), IOUs


def get_gt_masks(gt_masks_path, TEXT_PROMPT, frames, size):

    path = os.path.join(gt_masks_path, 'SegmentationClass')
    gt_masks = []
    for frame_pil in frames:

        # load image as array
        frame_mask_name = frame_pil.replace('jpg', 'png')
        try:
            gt_mask = Image.open(os.path.join(path, frame_mask_name))
            gt_mask = np.array(gt_mask)

            # select class
            path2label = os.path.join(gt_masks_path, 'labelmap.txt')

            # define labels
            with open(path2label) as file:
                data = file.read().split('\n')

            row_labels = [row_lable.split(':') for row_lable in data[1:]]
            labels = {label[0]:eval(label[1]) for label in row_labels if len(label)> 1}

            # create final masks
            final_mask = []
            for ind, val in enumerate(labels[TEXT_PROMPT]):
                mask = gt_mask[..., ind] == val
                final_mask.append(mask[..., np.newaxis])
                
            final_mask = np.concatenate(final_mask, axis=2).mean(axis=2)
            final_mask = (final_mask == 1).astype(int)


        except FileNotFoundError: 
            gt_mask = np.zeros(size)

        gt_masks.append(final_mask)
    
    return gt_masks


