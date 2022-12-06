"""
Adapted from https://github.com/TwentyBN/GulpIO-benchmarks
"""

from __future__ import print_function, division
import os
import glob
import sys
import time
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pdb

from pprint import pprint

import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision.transforms import *
from utils.torch_videovision.videotransforms import video_transforms, volume_transforms

# from gulpio import GulpDirectory
# from .gulp_data_parser import GulpDataset
from .misc import save_images_for_debug

# gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
# folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
# folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
# trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"

gulp_videos_path = "./data/gulped_rhexis/"
# folds_txt_path = "./data/folds4/{}/"
folds_txt_path = "./data/April2019/folds4/"
folds_txt_path_dum = "./data/dum_folds/"
trajectory_path = "./data/rhexis_tool_features/"


def prepare_split(config, labels_path=None, val_fold=3, test_fold=4, num_folds=4):
    if not labels_path:
        labels_path=folds_txt_path
    if '{}' in labels_path:
        labels_path = labels_path.format(config.labels_path)
    
    if (val_fold<1 or val_fold>num_folds or test_fold<1 or test_fold>num_folds):
        raise IndexError('Invalid fold id')
    all_folds = list(range(1,num_folds+1))
    train_folds = [f for f in all_folds if (f != val_fold and f!= test_fold)]
    
    print("Loading folds from {}".format(labels_path))
    train_ids = []
    train_labels = []
    for tf in train_folds:
        temp = pd.read_csv(labels_path + 'fold_{}.txt'.format(tf), header=0)
        train_labels.extend(temp['Label'].tolist())
        temp = temp['Dir'].str.split(" ", n = 1, expand = True)
        train_ids.extend(temp[0].tolist())
                
    temp = pd.read_csv(labels_path + 'fold_{}.txt'.format(val_fold), header=0)
    valid_labels = temp['Label'].tolist()
    temp = temp['Dir'].str.split(" ", n = 1, expand = True)
    valid_ids = temp[0].tolist()
        
    temp = pd.read_csv(labels_path + 'fold_{}.txt'.format(test_fold), header=0)
    test_labels = temp['Label'].tolist()
    temp = temp['Dir'].str.split(" ", n = 1, expand = True)
    test_ids = temp[0].tolist()

    train_df = pd.DataFrame({0: train_ids, 1: train_labels})
    print(train_df)
    valid_df = pd.DataFrame({0: valid_ids, 1: valid_labels})
    test_df = pd.DataFrame({0: test_ids, 1: test_labels})
        
    return train_df, valid_df, test_df

class VideoFolder(data.Dataset):

    def __init__(self, root, input, clip_size,
                 nclips, step_size, is_val, transform=None,):
        """
        Parameters:
            root: gulp data directory
            input: info about videos to include in the dataloader
            clip_size: number of frames in each example
            nclips: number of clips (each of clip_size). If -1, full video is used
            step_size: distance between consecutive frames
            is_val: is validation loader
            transform: transforms for the frame images
        """
        #if not os.path.exists(root):
        #    raise FileNotFoundError("No such folder: {}".format(root))

#         self.dataset_object = GulpDataset(input, csv=False)
        self.csv_data = input
        self.classes_dict = {'Nonexpert': 0, 0: 'Nonexpert', 'Expert': 1, 1: 'Expert'}
        self.classes = ['Nonexpert', 'Expert']

#         self.gulp_directory = GulpDirectory(root)
#         self.merged_meta_dict = self.gulp_directory.merged_meta_dict

        self.transform = transform

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.num_frames_array = []
        self._get_num_frames()

    def _get_num_frames(self):
        self.num_videos = len(self.csv_data)
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.csv_data.loc[i, 0]
            num_frames = len(glob.glob(path + "*.png"))
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)

    def __getitem__(self, index, test_pos=-1):
        item_path = self.csv_data.loc[index, 0] #Dir
        item_label = self.csv_data.loc[index, 1] #Label
        # num_frames = len(glob.glob(item_path + "*.jpg"))
        num_frames = self.num_frames_array[index]
        # print('num_frames: ' + str(num_frames) + ' in vid: ' + str(item_path))
        assert num_frames > 0
        # print(num_frames)
        # target_idx = self.classes_dict[int(item.label)]
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips
            # num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 1


        # old method
        # if num_frames_necessary < num_frames:
        #     # If there are more frames, then sample starting offset.
        #     diff = (num_frames - num_frames_necessary)
        #     # temporal augmentation
        #     # if not self.is_val:
        #     #     offset = np.random.randint(0, diff)

        #     offset = np.random.randint(0, diff)
        #     # offset = 10

        # if test_pos >= 0:
        #     offset = test_pos * self.clip_size * self.step_size // 2
        #     if offset > num_frames:
        #         return None
        #     print(offset)

        # slice_object = slice(
        #     offset, num_frames_necessary + offset, self.step_size)
        
        # indices = np.arange(num_frames)[slice_object]




        # divid into 8 part, get 32 frames inside of each part
        indices = []
        if num_frames_necessary < num_frames:
            # Divide whole video in to num_parts parts.
            num_parts = 8
            
            frames_per_part = int(self.clip_size /  num_parts)# 32
            assert (self.clip_size % num_parts) == 0,'num_parts * frames_per_part shold equals to clip_size'
            
            for i in range(num_parts):
                start = np.floor(i / num_parts * num_frames).astype(int) + 1 # avoid 0000.png
                end = np.floor((i + 1) / num_parts * num_frames).astype(int) + 1 # avoid 0000.png
                random_select = np.sort(np.random.randint(start,end,frames_per_part)) # np.random.randint, low inclusive, high exclusicve
                indices.append(random_select)
            indices_flat = np.array((indices)).flatten()
        frames = []
        for img_idx in indices_flat:
            item_name = item_path.split('/')[-2]
            zidx = str(img_idx).zfill(4)
            img_name = item_name + '_' + str(zidx)
            fpath = f"{item_path}{img_name}.png"
            assert os.path.exists(fpath), fpath
            frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
            frames.append(frame)
        # print("DEBUG:  {} frames had been loaded with cv2dataloader".format(len(frames)))
#         print(frames[0].shape)
#         print(frames.shape, frames.dtype)
#         assert 0 == 1
        if len(frames) < (self.clip_size * self.nclips):
            frames.extend([frames[-1]] *
                          ((self.clip_size * self.nclips) - len(frames)))
        # imgs = []
        # for img in frames:
        #     img = self.transform(img)
        #     imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        # data = torch.cat(imgs)
        # data = data.permute(1, 0, 2, 3)     # [3, frames, imsize, imsize]

        data  = self.transform(frames)
        # print(data.shape)

        # print(item.id, data.shape)
#         try:
#             vid_traj_path = trajectory_path + '{}.txt'.format(item.id)
#             vid_traj = torch.tensor(np.array(pd.read_csv(vid_traj_path, header=None)))
#             vid_traj = vid_traj.permute(1, 0)
#         except FileNotFoundError:
#             print('Could not find file ', vid_traj_path)
#             vid_traj = torch.tensor([0])
        # print(vid_traj.shape, vid_traj)

        return (data, target_idx, 0, item_path, frames)

    def __len__(self):
        return len(self.csv_data)


def get_data_loaders(config, val_fold=4, test_fold=5, imsize=84, batch_size=2, clip_size=18, 
                        nclips=1, step_size=2, num_workers=0, dum=False, num_folds=5):

    # transform = Compose([
    #                     ToPILImage(),
    #                     Resize(imsize),
    #                     CenterCrop(imsize),
    #                     ToTensor(),
    #                     # Normalize(mean=[0.485, 0.456, 0.406],
    #                     #           std=[0.229, 0.224, 0.225])
    #                     ])
    video_transform_list = [
                            video_transforms.RandomRotation(30),
                            video_transforms.RandomHorizontalFlip(),
                            video_transforms.Resize(imsize),
                            video_transforms.RandomCrop(imsize),
                            volume_transforms.ClipToTensor()
                            ]
    transform = video_transforms.Compose(video_transform_list)

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, test_df = prepare_split(config=config, labels_path=fpath, val_fold=val_fold, test_fold=test_fold, 
                                      num_folds=num_folds)

    train_loader = VideoFolder(root=gulp_videos_path,
                         input=train_df,
                         clip_size=clip_size,
                         nclips=nclips,
                         step_size=step_size,
                         is_val=False,
                         transform=transform,
                         )

    val_video_transform_list = [
                            video_transforms.Resize(imsize),
                            video_transforms.CenterCrop(imsize),
                            volume_transforms.ClipToTensor()
                            ]
    val_transform = video_transforms.Compose(val_video_transform_list)

    valid_loader = VideoFolder(root=gulp_videos_path,
                         input=valid_df,
                         clip_size=clip_size,
                         nclips=nclips,
                         step_size=step_size,
                         is_val=True,
                         transform=val_transform,
                         )

    test_loader = VideoFolder(root=gulp_videos_path,
                         input=test_df,
                         clip_size=clip_size,
                         nclips=nclips,
                         step_size=step_size,
                         is_val=True,
                         transform=val_transform,
                         )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(train_loader,
                            batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_loader,
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_loader,
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader
