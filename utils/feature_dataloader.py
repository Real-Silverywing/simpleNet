"""
Adapted from https://github.com/TwentyBN/GulpIO-benchmarks
"""

from __future__ import print_function, division
import os
import re
import glob
import time
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pprint import pprint

import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

# from utils.torch_videovision.videotransforms import (
#     video_transforms,
#     volume_transforms,
# )
from .cv2dataloader import prepare_split

from .misc import save_images_for_debug

error_vid_frame = {
    "./data/dataset99/image_extracts/vid_313_ffmpegscale/": [643],
    "/scratch/groups/svedula3/data/cataract/dataset99/image_extracts/vid_313_ffmpegscale/": [
        643
    ],
}

gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"

feature_root_path = "./data/processed_vids/{}/test_{}/"


def get_vid_feat_npy_from_path(path):
    return os.path.basename(os.path.normpath(path)) + ".npy"


class FeatureDataset(data.Dataset):
    def __init__(
        self,
        root_path,
        input,
        clip_size,
        step_size,
        decoder_dim=1024,
        model_type="TCN",
        rotate_augment=True,
        force_subsample=False,
    ):
        self.force_subsample = force_subsample
        self.rotate_augment = rotate_augment
        self.model_type = model_type
        self.clip_size = clip_size
        self.decoder_dim = decoder_dim
        self.csv_data = input
        self.classes_dict = {
            "Nonexpert": 0,
            0: "Nonexpert",
            "Expert": 1,
            1: "Expert",
        }
        self.classes = ["Nonexpert", "Expert"]
        self.step_size = step_size
        self.root_path = root_path
        self.num_frames_array = []
        self.get_num_frames()
        # self.pre_load_npys = {}

    def _get_npy_path(self, index):
        item_path = self.csv_data.loc[index, 0]  # Dir
        item_path = os.path.join(
            self.root_path, get_vid_feat_npy_from_path(item_path)
        )
        return item_path

    # def _preload(self):
    #     for i in range(len(self.csv_data)):
    #         item_path = self._get_npy_path(i)

    #         self.pre_load_npys[item_path] =

    def get_num_frames(self):
        self.num_videos = len(self.csv_data)
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.csv_data.loc[i, 0]
            num_frames = len(glob.glob(path + "*.jpg"))
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)

    def _transform(self, feat):
        # feat.shape = (N,D)
        N, _ = feat.shape
        if self.rotate_augment:
            rand_shift = torch.randint(0, N, (1,)).item()
            return np.roll(feat, shift=rand_shift, axis=0)
        else:
            return feat

    def _pack_feat(self, feat):
        F, D = feat.shape
        if self.model_type == "TCN" or self.force_subsample:
            if F < self.clip_size:
                # feat = np.resize(feat, (self.clip_size,D))
                feat = np.pad(feat, ((0, self.clip_size - F), (0, 0)))
                # print("in _pack_feat, F<self.clip_size, type: {}".format(type(feat)))
            elif F > self.clip_size:
                diff = F - self.clip_size
                offset = torch.randint(0, diff, (1,)).item()
                feat = feat[offset : self.clip_size + offset, :]
                # print("in _pack_feat, F>self.clip_size, type: {}".format(type(feat)))
            return feat
        elif self.model_type == "LSTM":
            return feat
        else:
            return feat

    def __getitem__(self, index):
        item_path = self._get_npy_path(index)
        # print("DEBUG: item_path: {}".format(item_path))
        num_frames = self.num_frames_array[index]
        feat = np.load(item_path, allow_pickle=True)
        F, D = feat.shape
        slice_object = slice(0, F, self.step_size)
        indices = torch.from_numpy(np.arange(F)[slice_object])
        feat = feat[slice_object]
        F, D = feat.shape
        feat = self._pack_feat(feat)
        feat = self._transform(
            feat
        )  # roll the feat along temporal dim (axis=0)

        feat = torch.from_numpy(feat).float()
        # print("DEBUG, feat.shape: {}".format(feat.shape))
        item_label = self.csv_data.loc[index, 1]  # Label
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        return feat, target_idx, item_path, torch.tensor([0]),indices  # place holder

    def __len__(self):
        return len(self.csv_data)


def collate(batch):
    return (
        pad_sequence([x[0] for x in batch], batch_first=True),
        torch.tensor([x[1] for x in batch]),
        [x[2] for x in batch],
        [x[0].shape[0] for x in batch],
        [x[4] for x in batch]
    )


def get_data_loaders(
    config,
    val_fold=4,
    test_fold=5,
    step_size=2,
    num_workers=0,
    dum=False,
    num_folds=5,
    shuffle_train=True,
):
    # only supports batchsize=1, because different videos have different length
    batch_size = config.batch_size

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=val_fold,
        test_fold=test_fold,
        num_folds=num_folds,
    )

    root_path = feature_root_path.format(config.processed_run, test_fold)

    train_loader = FeatureDataset(
        root_path=root_path,
        input=train_df,
        clip_size=config.clip_size,
        step_size=step_size,
        model_type=config.model_type,
        rotate_augment=True,
        force_subsample=config.force_subsample,
    )

    valid_loader = FeatureDataset(
        root_path=root_path,
        input=valid_df,
        clip_size=config.clip_size,
        step_size=step_size,
        model_type=config.model_type,
        rotate_augment=False,
        force_subsample=config.force_subsample,
    )

    test_loader = FeatureDataset(
        root_path=root_path,
        input=test_df,
        clip_size=config.clip_size,
        step_size=step_size,
        model_type=config.model_type,
        rotate_augment=False,
        force_subsample=config.force_subsample,
    )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    if config.model_type == "LSTM" and batch_size > 1:
        collate_fn = collate
    else:
        collate_fn = None

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader
