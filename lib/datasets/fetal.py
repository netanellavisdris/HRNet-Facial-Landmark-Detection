# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.mixture import GaussianMixture

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

curidx = 0
out_dir = "/tmp/OutTrain"
os.makedirs(out_dir, exist_ok=True)

class FetalLandmarks(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.reassign = cfg.TRAIN.REASSIGN
        self.anatomy = cfg.DATASET.ANATOMY

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file, header=0, sep=',')
        self.landmarks_frame.drop(self.landmarks_frame.columns[0], axis=1, inplace=True)

        # Remove rows where any landmarks are negative
        if self.anatomy == 'brain' or self.anatomy == 'abdomen':
            landmark_cols = self.landmarks_frame.columns[8:12]
        else:
            landmark_cols = self.landmarks_frame.columns[4:8]

        mask = (self.landmarks_frame[landmark_cols] < 0).any(axis=1)
        self.landmarks_frame = self.landmarks_frame[~mask].reset_index(drop=True)

        if is_train:
            if self.anatomy == 'brain' or self.anatomy == 'abdomen':
                landmarks = np.array(self.landmarks_frame.iloc[1:, 8:12].values, dtype=np.float32)
                print(landmarks)
                landmarks = landmarks.reshape(-1, 2)
                self.d_vect = determine_direction(landmarks)
            else:
                landmarks = np.array(self.landmarks_frame.iloc[1:, 4:8].values, dtype=np.float32)
                landmarks = landmarks.reshape(-1, 2)
                self.d_vect = determine_direction(landmarks)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        
        # idx += 1
        
        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        if self.anatomy == 'brain' or self.anatomy == 'abdomen':
            pts = self.landmarks_frame.iloc[idx, 8:12].values
        else:
            pts = self.landmarks_frame.iloc[idx, 4:8].values
        
        pts = pts[[1,0,3,2]]
        pts = pts.astype('float').reshape(-1, 2)

        # Clip the points 
        pts = np.clip(pts, a_min=0, a_max=None)

        # print(f'pts: {pts}')
        
        scale *= 1.7
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='FETAL')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] >= 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                # print(f'tpts[i, 0:2]: {tpts[i, 0:2]}')
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
            else:
                print ("ERROR HERE!!!!!")
        
        # TODO: Remove
        if self.is_train:
            if self.reassign :
                # if tpts[0,0] > tpts[1,0]:
                #     # print ('Right')
                #     pass
                # else:
                #     # print('Left')
                #     tmp = tpts[0, :].copy()
                #     tpts[0, :] = tpts[1, :]
                #     tpts[1, :] = tmp
                tpts = classify_direction(tpts, self.d_vect)
                tpts = tpts.reshape(-1, 2)
                target = np.zeros((nparts, self.output_size[0], self.output_size[1])) 
                for i in range(nparts):
                    target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                                label_type=self.label_type)
        img = img.astype(np.float32)

        newimg = img.copy()
        # print(f'newimg.shape: {newimg.shape}')
        # newimg[:, :, 1] = scipy.misc.imresize(target[0], (256, 256))
        # newimg[:, :, 2] = scipy.misc.imresize(target[1], (256, 256))
        newimg[:, :, 1] = cv2.resize(target[0]*255, (256, 256), interpolation=cv2.INTER_LINEAR)
        newimg[:, :, 2] = cv2.resize(target[1]*255, (256, 256), interpolation=cv2.INTER_LINEAR)
        newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        # cv2.imwrite(os.path.join(out_dir, "img{}_ttindex{}.png".format(idx, curidx)), newimg)
        globals()["curidx"] = curidx + 1
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta

def determine_direction(pts_arr, do_plot = True):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(pts_arr)
    if  do_plot:
        plt.scatter(pts_arr[::2,0], pts_arr[::2,1], alpha=1)
        plt.scatter(pts_arr[1::2,0], pts_arr[1::2,1], color='r', alpha=1)
        plt.plot(gmm.means_[:,0], gmm.means_[:,1], color='k')
        # save plot
        plt.savefig('/tmp/OutTrain/plot.png')
    
    return gmm.means_

def classify_direction(pts_arr, d_pts):
    d_vector = d_pts[1,:] - d_pts[0,:]
    ap = pts_arr
    pt_part1 = np.dot(d_vector[np.newaxis,:], ap.T) / np.linalg.norm(d_vector)
    pts_class = pt_part1.flatten()
    pts_class = pts_class.reshape(-1, 2)
    pts_class = np.stack((np.argmin(pts_class, axis=1), np.argmax(pts_class, axis=1)), axis=1)
    
    pts_remap = pts_class + (np.arange(pts_class.shape[0]) * 2)[:, np.newaxis]
    ret_pts = pts_arr[pts_remap]
    
    return ret_pts

if __name__ == '__main__':

    pass
