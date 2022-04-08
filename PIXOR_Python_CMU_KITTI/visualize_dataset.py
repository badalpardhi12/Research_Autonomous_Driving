import os.path
import numpy as np
import time
import torch
from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn
import cv2
import matplotlib.pyplot as plt
import math
import json
import os.path
from PIL import Image
import sys
import os
from data_processor.cmu_datagen import get_cmu_data_loader
from utils import load_config
import skvideo.io

def visualize():
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param map_height: height of the map
    :param window_name: name of the open_cv2 window
    :return: None
    '''
    config_name='config.json'
    config, _, _, _ = load_config(config_name)
    loader, _ = get_cmu_data_loader(batch_size=1, use_npy=config['use_npy'], frame_range=config['frame_range'])

    images = []

    print("\n\n Loading data...")
    for image_id in tqdm(range(loader.dataset.__len__())):
        pc_feature, label_map = loader.dataset[image_id]
        pc_feature = pc_feature.numpy()
        img = loader.dataset.get_image(image_id)
        image = np.array(img, dtype=np.uint8)

        intensity = np.zeros((pc_feature.shape[0], pc_feature.shape[1], 3))

        # print(pc_feature[::-1, :, :-1].shape)
        # print(pc_feature[::-1, :, :-1].max(axis=2).shape)
        # print(intensity[:, :, 0].shape)

        val = 1 - pc_feature[::-1, :, :-1].max(axis=2)
        intensity[:, :, 0] = val
        intensity[:, :, 1] = val
        intensity[:, :, 2] = val
        intensity = np.uint8(intensity * 255)
        intensity = cv2.rotate(intensity, cv2.ROTATE_90_COUNTERCLOCKWISE)
        intensity = cv2.flip(intensity, 1)

        bev_height, bev_width = intensity.shape[0], intensity.shape[1]
        resized_image = cv2.resize(image, (bev_width, bev_height), interpolation=cv2.INTER_AREA)

        final_image = np.concatenate((resized_image, intensity), axis=1)
        cv2.imwrite('Images/frame_' + str(image_id) + '.png', final_image)
        images.append(final_image)

    print("\n\n converting to video.....")
    skvideo.io.vwrite("outputvideo.mp4", images)
    print("\n\n video saved! \n\n")

if __name__ == '__main__':
    visualize()