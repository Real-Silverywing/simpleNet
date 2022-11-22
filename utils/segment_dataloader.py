"""
Adapted from https://github.com/TwentyBN/GulpIO-benchmarks
"""

from __future__ import print_function, division
import os
import glob
import sys
import time
from numpy.core.fromnumeric import size
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
from numpy.random import default_rng
from .cv2dataloader import prepare_split

# from torchvision.transforms import *
from utils.torch_videovision.videotransforms import video_transforms, volume_transforms

# from gulpio import GulpDirectory
# from .gulp_data_parser import GulpDataset
from .misc import save_images_for_debug

gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"

# gulp_videos_path = "./data/gulped_rhexis/"
# folds_txt_path = "./data/folds4/{}/"
# folds_txt_path_dum = "./data/dum_folds/"
# trajectory_path = "./data/rhexis_tool_features/"

class SegmentVideoFolder(data.Dataset):
    def __init__(
        self, input, num_segments, size_snippets, is_val, transform=None,
    ):
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
        # if not os.path.exists(root):
        #    raise FileNotFoundError("No such folder: {}".format(root))

        #         self.dataset_object = GulpDataset(input, csv=False)
        self.csv_data = input
        self.classes_dict = {"Nonexpert": 0, 0: "Nonexpert", "Expert": 1, 1: "Expert"}
        self.classes = ["Nonexpert", "Expert"]

        #         self.gulp_directory = GulpDirectory(root)
        #         self.merged_meta_dict = self.gulp_directory.merged_meta_dict

        self.transform = transform

        self.num_segments = num_segments
        self.size_snippets = size_snippets
        self.is_val = is_val
        self.num_frames_array = []
        self.error_vid_frame = {
            "./data/dataset99/image_extracts/vid_313_ffmpegscale/": [643]
        }
        self._get_num_frames()

    def _get_num_frames(self):
        self.num_videos = len(self.csv_data)
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.csv_data.loc[i, 0]
            num_frames = len(glob.glob(path + "*.jpg"))
            if path in self.error_vid_frame.keys():
                num_frames -= len(self.error_vid_frame.keys())
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)

    def read_images(self, idx):
        item_path = self.csv_data.loc[idx, 0]  # Dir
        num_frames = self.num_frames_array[idx]
        if self.num_segments >= 1:
            num_frames_necessary = self.num_segments * self.size_snippets
        else:
            num_frames_necessary = num_frames

        if num_frames_necessary > num_frames:
            diff = num_frames_necessary-num_frames
            frames = []
            ind = 0
            # print("In segment dataloader, read_img, no enough frames, ind: ", ind)
            while ind<num_frames:
                if item_path in self.error_vid_frame.keys() and ind in self.error_vid_frame[item_path]:
                    ind+=1
                fpath = f"{item_path}{ind}.jpg"
                assert os.path.exists(fpath), fpath
                frames.append(cv2.imread(fpath))
                ind+=1
            # print("In segment dataloader, read_img, no enough frames, len(frames): ", len(frames))
            imgs = self.transform(frames)
            # padd 0 for videos that is so short
            imgs = torch.cat((imgs,torch.zeros((imgs.shape[0],diff,*imgs.shape[2:]))),dim=1)
        else:
            segment_len = self.num_frames_array[idx] // self.num_segments
            rng = default_rng()
            frames = []
            # print("In segment dataloader, read_img, segment, segment_len: ", segment_len)
            for i in range(self.num_segments):
                slice = np.sort(
                    rng.choice(range(segment_len), self.size_snippets, replace=False)
                )
                for j in slice:
                    ind = i*self.size_snippets+j

                    # I assume the error img is not the last image, because if 
                    # so, we can simply delete that image from the dataset
                    if item_path in self.error_vid_frame.keys() and ind in self.error_vid_frame[item_path]:
                        ind+=1
                    fpath = f"{item_path}{ind}.jpg"
                    assert os.path.exists(fpath), fpath
                    frames.append(cv2.imread(fpath))
            
            # print("In segment dataloader, read_img, segment, len(frames): ", len(frames))
            imgs = self.transform(frames)
            if imgs.shape[1]!=self.num_segments*self.size_snippets:
                print("DEBUG: imgs.shape[0]!=self.num_segments*self.size_snippets: ", imgs.shape)
                print("At video: ", item_path)
        return imgs, frames

    def __getitem__(self, index, test_pos=-1):
        item_path = self.csv_data.loc[index, 0]  # Dir
        item_label = self.csv_data.loc[index, 1]  # Label
        num_frames = self.num_frames_array[index]
        assert num_frames > 0
        # print(num_frames)
        # target_idx = self.classes_dict[int(item.label)]
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        # print("DEBUG: before read_images")
        data, frames = self.read_images(index)
        # print("DEBUG: data.shape in __getitem__",data.shape)

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


def get_data_loaders(
    config,
    val_fold=4,
    test_fold=5,
    batch_size=2,
    imsize=256,
    num_segments=18,
    size_snippets=1,
    num_workers=0,
    dum=False,
    num_folds=5,
):

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
        volume_transforms.ClipToTensor(),
    ]
    transform = video_transforms.Compose(video_transform_list)

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=val_fold,
        test_fold=test_fold,
        num_folds=num_folds,
    )

    train_loader = SegmentVideoFolder(
        input=train_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=False,
        transform=transform,
    )

    val_video_transform_list = [
        video_transforms.Resize(imsize),
        video_transforms.CenterCrop(imsize),
        volume_transforms.ClipToTensor(),
    ]
    val_transform = video_transforms.Compose(val_video_transform_list)

    valid_loader = SegmentVideoFolder(
        input=valid_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=True,
        transform=val_transform,
    )

    test_loader = SegmentVideoFolder(
        input=test_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=True,
        transform=val_transform,
    )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader
