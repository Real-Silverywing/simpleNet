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

from numpy.random import default_rng
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from utils.torch_videovision.videotransforms import video_transforms, volume_transforms
from .cv2dataloader import prepare_split

from .misc import save_images_for_debug

gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"
preprocessed_data_path = "../data/image_extract/"


class VideoDataset(data.Dataset):
    def __init__(
        self,
        input,
        num_segments,
        size_snippets,
        is_val,
        preprocessed_data_path,
    ):
        """
        Parameters:
            root: gulp data directory
            input: info about videos to include in the dataloader
            clip_size: number of frames in each example
            nclips: number of clips (each of clip_size). If -1, full video is used
            is_val: is validation loader
        """
        # if not os.path.exists(root):
        #    raise FileNotFoundError("No such folder: {}".format(root))

        #         self.dataset_object = GulpDataset(input, csv=False)
        self.preprocessed_data_path = preprocessed_data_path
        self.csv_data = input
        self.classes_dict = {"Nonexpert": 0, 0: "Nonexpert", "Expert": 1, 1: "Expert"}
        self.classes = ["Nonexpert", "Expert"]

        #         self.gulp_directory = GulpDirectory(root)
        #         self.merged_meta_dict = self.gulp_directory.merged_meta_dict

        self.num_segments = num_segments
        self.size_snippets = size_snippets
        self.is_val = is_val
        self.file_name_place_holder = self.preprocessed_data_path + "{}.npy"
        self.num_frames_array = []
        self.num_videos = -1
        self.error_vid_frame = {
            "./data/dataset99/image_extracts/vid_313_ffmpegscale/": [643]
        }
        self.extracted_shape = (2048, 8, 8)
        self.get_num_frames()

    def get_num_frames(self):
        self.num_videos = len(self.csv_data)
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.csv_data.loc[i, 0]
            num_frames = len(glob.glob(path + "*.png"))
            if path in self.error_vid_frame.keys():
                num_frames -= len(self.error_vid_frame.keys())
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)

    def get_vid_from_path(self, path):
        return os.path.basename(os.path.normpath(path))

    def _sample(self, idx, dataset):
        segment_len = self.num_frames_array[idx] // self.num_segments
        sampled_frames = np.zeros(
            (self.num_segments * self.size_snippets, *self.extracted_shape)
        )
        rng = default_rng()

        for i in range(self.num_segments):
            slice = np.sort(
                rng.choice(range(segment_len), self.size_snippets, replace=False)
            )
            sampled_frames[
                i * self.size_snippets : (i + 1) * self.size_snippets, :, :, :
            ] = dataset[slice, :, :, :]

        return sampled_frames

    def __getitem__(self, index):
        item_path = self.csv_data.loc[index, 0]  # Dir
        item_label = self.csv_data.loc[index, 1]  # Label
        num_frames = self.num_frames_array[index]
        dataset = np.memmap(
            self.file_name_place_holder.format(
                self.get_vid_from_path(self.csv_data.loc[index, 0])
            ),
            np.float32,
            "w+",
            shape=(num_frames, *self.extracted_shape),
        )
        print("num_frames: " + str(num_frames) + " in vid: " + str(item_path))
        assert num_frames > 0
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        if self.num_segments >= 1:
            num_frames_necessary = self.num_segments * self.size_snippets
        else:
            num_frames_necessary = num_frames

        if num_frames_necessary > num_frames:
            diff = num_frames_necessary-num_frames
            data = np.copy(dataset)
            data = np.concatenate((data,np.zeros((diff,*data.shape[1:]))),axis=0)
        else:
            data = self._sample(index, dataset)

        data = torch.from_numpy(data)
        if not isinstance(data, torch.FloatTensor):
            data = data.float()

        return (data, target_idx, 0, item_path)

    def __len__(self):
        return len(self.csv_data)


def get_data_loaders(
    config,
    val_fold=4,
    test_fold=5,
    batch_size=2,
    num_segments=18,
    size_snippets=1,
    num_workers=0,
    dum=False,
    num_folds=5,
):

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=val_fold,
        test_fold=test_fold,
        num_folds=num_folds,
    )

    train_loader = VideoDataset(
        input=train_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=False,
        preprocessed_data_path=preprocessed_data_path,
    )

    valid_loader = VideoDataset(
        input=valid_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=True,
        preprocessed_data_path=preprocessed_data_path,
    )

    test_loader = VideoDataset(
        input=test_df,
        num_segments=num_segments,
        size_snippets=size_snippets,
        is_val=True,
        preprocessed_data_path=preprocessed_data_path,
    )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, valid_loader, test_loader
