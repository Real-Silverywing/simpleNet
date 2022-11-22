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
from tokenize import group
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

# from torchvision.transforms import *
from utils.torch_videovision.videotransforms import (
    video_transforms,
    volume_transforms,
)
from .cv2dataloader import prepare_split

# from gulpio import GulpDirectory
# from .gulp_data_parser import GulpDataset
from .misc import save_images_for_debug

error_vid_frame = {
    "./data/dataset99/image_extracts/vid_313_ffmpegscale/": [643],
    "./data/cohort2021/vid_953/": [0],
    "./data/cohort2021/vid_919/": [0],
}

TEST_CSV_PATH = "./data/new_all_labels.csv"  # Made using prepare_test_set.py

EXT_DATASET_LEN = 51
included_ext_samples = [0, 1, 2, 3, 6, 7, 9, 14, 15, 38]


# gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
# folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
# folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
# trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"

gulp_videos_path = "./data/gulped_rhexis/"
folds_txt_path = "./data/folds4/{}/"
folds_txt_path_dum = "./data/dum_folds/"
trajectory_path = "./data/rhexis_tool_features/"

feature_root_path = "./data/processed_vids/run38/{}/temp_last/"

SPEED_RATIO = 2.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, optimizer, model_ckpt):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0

    print("Resuming from checkpoint '{}'".format(model_ckpt))

    if os.path.isfile(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val_loss = checkpoint["best_val_loss"]
        print(
            "Loaded checkpoint '{}' (epoch {})".format(model_ckpt, checkpoint["epoch"])
        )

        model = model.to(device)
        model.train()
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    else:
        print("=> no checkpoint found at '{}'".format(model_ckpt))

    return model, optimizer, start_epoch, best_val_loss


class MixedDataset(data.Dataset):
    def __init__(
        self,
        input,
        nclips,
        clip_size,
        step_size,
        save_root,
        val_fold,
        included_ext_samples,
        use_feature=False,
        transform=None,
        random_interval=False,
        use_group_label=False,
        encoder_train=False,
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
        self.encoder_train = encoder_train
        self.use_feature = use_feature
        self.val_fold = val_fold
        self.save_root = save_root
        self.use_grop_label = use_group_label
        self.csv_data = input
        self.included_ext_samples = included_ext_samples
        self.ext_data = self.get_external_dataset_df()
        if self.use_grop_label:
            self.group_label_df = None
        self.classes_dict = {
            "Nonexpert": 0,
            0: "Nonexpert",
            "Expert": 1,
            1: "Expert",
        }
        self.classes = ["Nonexpert", "Expert"]
        self.nclips = nclips
        self.step_size = step_size

        #         self.gulp_directory = GulpDirectory(root)
        #         self.merged_meta_dict = self.gulp_directory.merged_meta_dict

        self.transform = transform

        self.clip_size = clip_size
        self.num_frames_array = []
        self.num_frames_dict = {}
        self._get_num_frames()
        self.random_interval = (step_size, step_size + 1)
        if random_interval:
            self.random_interval = (step_size, 30)
        # self.kp_transform = kp_transform
        if self.use_grop_label:
            self.group_label_df = self.get_group_label_data_df()
    
    def get_test_df(self):
        return self.ext_data

    def get_external_dataset_df(self):
        temp = pd.read_csv(TEST_CSV_PATH, header=0)
        test_labels = temp["Label"].tolist()
        temp = temp["Dir"].str.split(" ", n=1, expand=True)
        test_ids = temp[0].tolist()
        test_df = pd.DataFrame({0: test_ids, 1: test_labels})
        test_df = test_df.loc[self.included_ext_samples]
        test_df = test_df.reset_index(drop=True)
        print(test_df)
        return test_df

    def get_min_max_step(self, idx, num_frames):
        if self.random_interval[0] == self.random_interval[1] - 1:
            return (self.random_interval[0], self.random_interval[1])
        if idx < len(self.csv_data):
            random_interval = (
                min(int(self.random_interval[0] * (num_frames / 8677)), 8),
                int(self.random_interval[1] * (num_frames / 8677)),
            )
        else:
            random_interval = (
                min(
                    int(self.random_interval[0] * SPEED_RATIO * (num_frames / 20790)), 8
                ),
                int(self.random_interval[1] * SPEED_RATIO * (num_frames / 20790)),
            )
        if random_interval[0] == random_interval[1]:
            random_interval[1] += 1
        return random_interval

    def get_data_given_idx(self, idx, key):
        if idx < len(self.csv_data):
            return self.csv_data.loc[idx, key]
        else:
            return self.ext_data.loc[idx - len(self.csv_data), key]

    def calculate_group_label(self, a_is_src, b_is_src, a_label, b_label):
        if a_is_src and b_is_src and a_label and b_label:
            return 0
        elif (
            ((a_is_src and not b_is_src) or (not a_is_src and b_is_src))
            and a_label
            and b_label
        ):
            return 1
        elif a_is_src and b_is_src and (a_label ^ b_label):
            return 2
        elif ((a_is_src and not b_is_src) or (not a_is_src and b_is_src)) and (
            a_label ^ b_label
        ):
            return 3
        else:
            return 4

    def get_group_label_data_df(self):
        a_video_idxs = []
        b_video_idxs = []
        a_labels = []
        b_labels = []
        group_labels = []
        length = len(self.csv_data) + len(self.ext_data)
        print("length: {}".format(length))
        for i in range(length):
            for j in range(length):
                a_label = self.get_data_given_idx(i, 1)
                b_label = self.get_data_given_idx(j, 1)
                group_label = self.calculate_group_label(
                    i < len(self.csv_data), j < len(self.csv_data), a_label, b_label
                )
                if (not self.encoder_train and group_label < 4) or (
                    self.encoder_train and (group_label == 1 or group_label == 3)
                ):
                    a_video_idxs.append(i)
                    b_video_idxs.append(j)
                    a_labels.append(a_label)
                    b_labels.append(b_label)
                    group_labels.append(group_label)

        group_label_df = pd.DataFrame(
            {
                0: a_video_idxs,
                1: b_video_idxs,
                2: group_labels,
                3: a_labels,
                4: b_labels,
            }
        )
        return group_label_df

    def _get_num_frames(self):
        self.num_videos = self.__len__()
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.get_data_given_idx(i, 0)
            num_frames = len(glob.glob(path + "*.jpg"))
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)
            self.num_frames_dict[path] = num_frames

    def _get_vid_id_from_path(self, vid_path):
        return int(re.search(r"^.*vid_(\d+)", vid_path)[1])

    def _set_random_seed(self, seed):
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)  # forbid hash random
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _transform(self, data):
        # data should have shape: (frame, C, H, W), data is a list of (C,H,W) np arrays

        if self.transform:
            replay = None
            F = len(data)
            H = 0
            W = 0
            for i in range(F):
                if i == 0:
                    transformed = self.transform(image=data[i])
                    replay = transformed["replay"]
                    data[i] = transformed["image"]
                    C, H, W = data[i].shape
                    new_data = torch.zeros((F, C, H, W))
                else:
                    transformed = A.ReplayCompose.replay(replay, image=data[i])
                    data[i] = transformed["image"]

                new_data[i] = data[i]
            return new_data
        else:
            return data

    def __getitem__(self, index):
        if self.use_grop_label:
            return self.getitem_group_label(index)
        else:
            return self.__getitem_single__(index)

    def get_cat_feature_or_calculate(self, a_idx, b_idx):
        feature_path_a = os.path.join(self.save_root, "{}.npy".format(a_idx))
        feature_path_b = os.path.join(self.save_root, "{}.npy".format(b_idx))
        feature_a = None
        feature_b = None
        if os.path.exists(feature_path_a):
            feature_a = np.load(feature_path_a)
        if os.path.exists(feature_path_b):
            feature_b = np.load(feature_path_b)
        return torch.from_numpy(feature_a).float(), torch.from_numpy(feature_b).float()

    def getitem_group_label(self, index):
        a_idx, b_idx, group_label, a_target_idx, b_target_idx = self.group_label_df.loc[
            index
        ]
        a_target_idx = torch.from_numpy(np.array(int(a_target_idx))).float()
        b_target_idx = torch.from_numpy(np.array(int(b_target_idx))).float()

        if self.use_feature:
            feature_a, feature_b = self.get_cat_feature_or_calculate(a_idx, b_idx)
            if feature_a is not None and feature_b is not None:
                return (
                    (feature_a, a_target_idx, a_idx,),
                    (feature_b, b_target_idx, b_idx,),
                    group_label,
                )

        (
            a_data,
            a_target_idx,
            a_is_ext,
            a_item_path,
            a_frames,
            a_indices,
            a_index,
        ) = self.__getitem_single__(a_idx)
        (
            b_data,
            b_target_idx,
            b_is_ext,
            b_item_path,
            b_frames,
            b_indices,
            b_index,
        ) = self.__getitem_single__(b_idx)

        return (
            (a_data, a_target_idx, a_is_ext, a_idx, a_item_path, a_frames, a_indices,),
            (b_data, b_target_idx, b_is_ext, b_idx, b_item_path, b_frames, b_indices,),
            group_label,
        )

    def __getitem_single__(self, index, test_pos=-1):
        num_frames = self.num_frames_array[index]
        item_path = self.get_data_given_idx(index, 0)
        item_label = self.get_data_given_idx(index, 1)
        # num_frames = len(glob.glob(item_path + "*.jpg"))
        # print('num_frames: ' + str(num_frames) + ' in vid: ' + str(item_path))
        assert num_frames > 0
        # print(num_frames)
        # target_idx = self.classes_dict[int(item.label)]
        random_interval = self.get_min_max_step(index, num_frames)
        # print("random_interval: {}".format(random_interval))
        min_step = random_interval[0]
        max_step = random_interval[1]
        step_size = np.random.randint(min_step, max_step)
        # print("step_size: {}".format(step_size))
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * step_size
        else:
            num_frames_necessary = num_frames
        offset = 0

        if test_pos >= 0:
            offset = test_pos * self.clip_size * step_size // 2
            if offset > num_frames:
                return None
            print(offset)

        frames = []
        indices = []
        for i in range(self.nclips):
            if num_frames_necessary < num_frames:
                # If there are more frames, then sample starting offset.
                diff = num_frames - num_frames_necessary
                # temporal augmentation
                # if not self.is_val:
                #     offset = np.random.randint(0, diff)

                # offset = np.random.randint(0, diff)
                offset = torch.randint(0, diff, (1,)).item()  # add seed
                # print("Kpdataloader, {} offset: {}".format(item_path, offset))

            slice_object = slice(
                offset, num_frames_necessary // self.nclips + offset, step_size,
            )

            clip_indices = np.arange(num_frames)[slice_object]
            # print("clip_indices: {}".format(clip_indices))

            #         frames, meta = self.gulp_directory[item.id, slice_object]
            for ind in clip_indices:
                fpath = f"{item_path}{ind}.jpg"  # Assuming no consecutive error imgs.
                # print("DEBUG: fpath: {}".format(fpath))
                try:
                    if (
                        item_path in error_vid_frame.keys()
                        and (ind) in error_vid_frame[item_path]
                    ):
                        print("DEBUG: encountered error vid: {}".format(fpath))
                        ind_ = ind - 1 if ind > 0 else ind + 1
                        fpath = f"{item_path}{ind_}.jpg"  # Assuming no consecutive error imgs.
                        frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
                    else:
                        # print("DEBUG: before read")
                        # frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
                        frame = cv2.imread(fpath)
                        # print("DEBUG: after read {}".format(frame))
                        # print("cv2.cvtColor: {}".format(cv2.cvtColor))
                        # print("cv2.COLOR_BGR2RGB: {}".format(cv2.COLOR_BGR2RGB))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # print("DEBUG: after cvt")
                except Exception as e:
                    print("Catched exception {}".format(e))
                    print("DEBUG: catched encountered error vid: {}".format(fpath))
                    print(
                        "item_path in keys: {}, img_idx in value: {}".format(
                            (item_path in error_vid_frame.keys()),
                            (ind in error_vid_frame[item_path])
                            if item_path in error_vid_frame.keys()
                            else "No",
                        )
                    )
                    ind_ = ind - 1 if ind > 0 else ind + 1
                    fpath = (
                        f"{item_path}{ind_}.jpg"  # Assuming no consecutive error imgs.
                    )
                    frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) < (self.clip_size):
                ori_len = len(frames)
                frames.extend([frames[-1]] * ((self.clip_size) - len(frames)))
                clip_indices = np.concatenate(
                    (
                        clip_indices,
                        np.repeat(clip_indices[-1], (self.clip_size - ori_len)),
                    )
                )
            indices.append(clip_indices)

        indices = np.concatenate(indices)
        # print("indices: {}".format(indices))

        data = self._transform(frames)
        # print("data: {}".format(data))
        # data.shape= (batch_size, num_frames, encoder_dim, h, w)

        return (
            data,
            target_idx,
            int(index > len(self.csv_data)),
            item_path,
            frames,
            indices,
            index,
        )

    def __len__(self):
        if self.use_grop_label and self.group_label_df is not None:
            return len(self.group_label_df)
        else:
            return len(self.csv_data) + len(self.ext_data)


def get_data_loaders(
    config,
    val_fold=4,
    included_ext_train_samples=[0, 1, 2, 3, 6, 7, 9, 14, 15, 38],
    included_ext_val_samples=[8, 10, 11, 12, 13, 16, 17, 18, 19, 20],
    imsize=84,
    batch_size=2,
    clip_size=18,
    nclips=1,
    step_size=2,
    num_workers=0,
    dum=False,
    num_folds=5,
    shuffle_train=True,
):

    cropsize = int(imsize * 0.875)

    transform = A.ReplayCompose(
        [
            A.Rotate(30),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(),
            A.SmallestMaxSize(max_size=imsize),
            A.RandomCrop(width=cropsize, height=cropsize),
            A.transforms.Normalize(),
            ToTensorV2(),
        ]
    )
    # kp_transform = video_transforms.Compose(video_transform_list[:-1]+(kp_post_transform_list))

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, _ = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=val_fold,
        test_fold=val_fold,
        num_folds=num_folds,
    )

    train_loader = MixedDataset(
        input=train_df,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        val_fold=val_fold,
        save_root=feature_root_path.format(val_fold),
        transform=transform,
        included_ext_samples=included_ext_train_samples
        if not config.exclude_ext_data
        else [],
        use_group_label=config.group_label,
        #  kp_transform = kp_transform
    )

    val_transform = A.ReplayCompose(
        [
            A.SmallestMaxSize(max_size=imsize),
            # A.Resize(height=imsize, width=imsize),
            A.CenterCrop(width=cropsize, height=cropsize),
            A.transforms.Normalize(),
            ToTensorV2(),
        ]
    )
    # val_transform = transform
    # val_kp_transform = video_transforms.Compose(val_video_transform_list[:-1]+(kp_post_transform_list))

    valid_loader = MixedDataset(
        input=valid_df,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        save_root=feature_root_path.format(val_fold),
        val_fold=val_fold,
        transform=val_transform,
        included_ext_samples=included_ext_val_samples
        if not config.exclude_ext_data
        else [],
        use_group_label=config.group_label,
        #  kp_transform = val_kp_transform
    )

    included_ext_test_samples = [
        i
        for i in range(EXT_DATASET_LEN)
        if (i not in included_ext_train_samples and i not in included_ext_val_samples)
    ]

    test_loader = MixedDataset(
        input=pd.DataFrame({0: [], 1: []}),
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        save_root=feature_root_path.format(val_fold),
        val_fold=val_fold,
        transform=val_transform,
        included_ext_samples=included_ext_test_samples,
        use_group_label=config.group_label,
        #  kp_transform = val_kp_transform
    )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=shuffle_train,
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
