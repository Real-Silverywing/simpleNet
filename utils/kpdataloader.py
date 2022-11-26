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

# from torchvision.transforms import *
# from utils.torch_videovision.videotransforms import (
#     video_transforms,
#     volume_transforms,
# )
from .cv2dataloader import prepare_split

# from gulpio import GulpDirectory
# from .gulp_data_parser import GulpDataset
from .misc import save_images_for_debug

error_vid_frame = {
    "./data/dataset99/image_extracts/vid_313_ffmpegscale/": [
        643
    ],
    "./data/cohort2021/vid_939/": [0],
}

# gulp_videos_path = "/scratch/groups/svedula3/data/cataract/gulped_rhexis/"
# folds_txt_path = "/scratch/groups/svedula3/data/cataract/folds4/{}/"
# folds_txt_path_dum = "/scratch/groups/svedula3/data/cataract/dum_folds/"
# trajectory_path = "/scratch/groups/svedula3/data/cataract/rhexis_tool_features/"

gulp_videos_path = "./data/gulped_rhexis/"
folds_txt_path = "./data/April2019/folds4/"
folds_txt_path_dum = "./data/dum_folds/"
trajectory_path = "./data/rhexis_tool_features/"


class KeyPointDataset(data.Dataset):
    def __init__(
        self,
        input,
        nclips,
        clip_size,
        step_size,
        is_val,
        transform=None,
        traj=False,
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
        self.traj = traj
        self.csv_data = input
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
        self.is_val = is_val
        self.num_frames_array = []
        self._get_num_frames()
        self.ANNO_COLS = [
            "frame_number",
            "tool_type",
            "p1x",
            "p1y",
            "p2x",
            "p2y",
            "p3x",
            "p3y",
            "p4x",
            "p4y",
            "p5x",
            "p5y",
            "p6x",
            "p6y",
        ]
        self.traject_dir = (
            "./data/kristen_processed/finalized/corrected_done/vid_{:03d}.txt"
        )
        self.vid_traj_df_dic = {}
        self._load_vid_traj_df_dic()
        # self.kp_transform = kp_transform
    
    def get_test_df(self):
        return self.csv_data

    def _get_num_frames(self):
        self.num_videos = len(self.csv_data)
        # print("DEBUG: number of videos: ", self.num_videos)
        for i in range(self.num_videos):
            path = self.csv_data.loc[i, 0]
            num_frames = len(glob.glob(path + "*.png"))
            print("num_frames: " + str(num_frames) + " in vid: " + str(path))
            self.num_frames_array.append(num_frames)

    def _get_vid_id_from_path(self, vid_path):
        return int(re.search(r"^.*vid_(\d+)", vid_path)[1])

    def _load_vid_traj_df_dic(self):
        num_videos = len(self.csv_data)
        for i in range(num_videos):
            path = self.csv_data.loc[i, 0]
            vid = self._get_vid_id_from_path(path)
            if not os.path.exists(self.traject_dir.format(vid)):
                self.vid_traj_df_dic[path] = pd.DataFrame()
                continue
            df = pd.read_csv(
                self.traject_dir.format(vid),
                header=None,
                index_col=0,
                names=self.ANNO_COLS,
            )
            df.index = df.index.astype(int)
            cur_type = None
            pre_index = None
            inter_idx = []
            data_idx = []
            cnt = 0
            test_df = df.copy()
            for index in df.index:
                cnt += 1
                t_type = df.loc[index, "tool_type"]

                # add idx to data_idx and inter_idx when cnt>=len(df) only
                if cnt >= len(df):
                    data_idx.append(index)
                    if pre_index != None:
                        inter_idx += list(range(pre_index + 1, index))
                        # for i in range(pre_index+1, index):
                        #     inter_idx.append(i)
                    pre_index = index

                if cur_type == None:
                    cur_type = t_type
                elif cur_type != t_type or cnt >= len(df):
                    if len(inter_idx) > 1:
                        #         print((np.array(inter_idx,dtype=np.int64)).dtype)
                        inter_head = np.stack(
                            (
                                np.array(inter_idx, dtype=np.int64),
                                np.tile(cur_type, (len(inter_idx))),
                            ),
                            axis=1,
                        )
                        inter_head[:, 0] = inter_head[:, 0].astype(np.int64)
                        # print(inter_head[:,0].dtype)
                        interpred = np.zeros((len(inter_idx), 12))
                        for p in range(12):
                            anno_pos_col = self.ANNO_COLS[2 + p]
                            interpred[:, p] = np.interp(
                                inter_idx,
                                data_idx,
                                df.loc[data_idx, anno_pos_col],
                            )
                        interpreted = np.concatenate(
                            (inter_head, interpred), axis=1
                        )
                        # print(interpreted.shape)
                        # print(self.ANNO_COLS)
                        # print(interpreted)
                        # print("VID: ", vid)
                        append_df = pd.DataFrame.from_records(
                            interpreted,
                            index="frame_number",
                            columns=self.ANNO_COLS,
                        )
                        append_df.index = append_df.index.astype(int)
                        for p in range(12):
                            anno_pos_col = self.ANNO_COLS[2 + p]
                            append_df[anno_pos_col] = append_df[
                                anno_pos_col
                            ].astype(float)
                        #         print(append_df.loc['61'])
                        test_df = test_df.append(append_df)
                    #         print(test_df)
                    cur_type = t_type
                    inter_idx = []
                    data_idx = []
                    pre_index = None

                data_idx.append(index)
                if pre_index != None:
                    inter_idx += list(range(pre_index + 1, index))
                    # for i in range(pre_index+1, index):
                    #     inter_idx.append(i)
                pre_index = index
                # cnt+=1
                # if cnt>10:
                #     break
            test_df = test_df.sort_index()
            test_df = test_df[test_df.tool_type != "no tools"]
            test_df = test_df.loc[:, test_df.columns != "tool_type"]
            self.vid_traj_df_dic[path] = test_df
            print(
                "vid: ", vid, "len(self.vid_traj_df_dic[path])=", len(test_df)
            )

    def _gaussian(self, x, mean, val):
        return (
            1
            / (val * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * np.square((x - mean) / val))
        )

    def get_traj_arrary(self, traj, indices, frames_needed):
        ret_traj = np.zeros((frames_needed, 12))
        len_indices = len(indices)
        t = 0
        for i in indices:
            if i not in traj.index:
                ret_traj[t] = -1
            else:
                ret_traj[t] = traj.loc[i].values
            t += 1
        if indices[len_indices - 1] not in traj.index:
            ret_traj[t:] = -1
        else:
            ret_traj[t:] = traj.loc[indices[len_indices - 1]].values
        ret_traj = np.reshape(ret_traj, (frames_needed, 6, 2))
        return ret_traj

    def _gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = torch.arange(0, width, 1, dtype=float)  ## (width,)
        y = torch.arange(0, height, 1, dtype=float).unsqueeze(1)  ## (height,1)
        return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def _traj_to_map_gaussian(self, traj, sample_frame, L=224, scale=3):
        C, h, w = sample_frame.shape
        down_scale = h // L
        unvalid_idx = traj < 0
        traj = (traj // down_scale).long()
        traj[unvalid_idx] = -1

        N, J, _ = traj.shape
        traj_map = torch.zeros((N, L, L, J))
        for i in range(N):
            for j in range(J):
                x, y = traj[i, j]
                if x < 0 or y < 0:
                    continue
                else:
                    gaussian_map = self._gaussian_k(x, y, scale, L, L)
                    traj_map[i, :, :, j] = gaussian_map

        return traj_map

    def _traj_to_map(self, traj, sample_frame, L=7, scale=1):
        # L: the side length of the map
        # traj.shape=(batch*frames,6,2)
        # sample_frame.shape= (encoder_dim, h, w)
        C, h, w = sample_frame.shape
        down_scale = h // L
        unvalid_idx = traj < 0
        traj = (traj // down_scale).long()
        traj[unvalid_idx] = -1

        N, J, _ = traj.shape
        traj_map = torch.zeros((N, L, L))
        # print("traj: ", traj.index)
        for i in range(N):
            for j in range(J):
                x, y = traj[i, j]
                if x < 0 or y < 0:
                    continue
                else:
                    x = int(x)
                    y = int(y)
                    x1 = max(x - scale, 0)
                    x2 = min(x + scale, w)
                    y1 = max(y - scale, 0)
                    y2 = min(y + scale, h)
                    traj_map[i, y1 : y2 + 1, x1 : x2 + 1] = 1
        return traj_map

    def _set_random_seed(self, seed):
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)  # forbid hash random
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _transform(self, data, traj):
        # traj should have shape: (frame,6,2)
        # data should have shape: (frame, C, H, W), data is a list of (C,H,W) np arrays

        traj = np.where(traj == -1, np.zeros_like(traj), traj)

        if self.transform:
            replay = None
            F = len(data)
            H = 0
            W = 0
            for i in range(F):
                if i == 0:
                    transformed = self.transform(
                        image=data[i], keypoints=traj[i].tolist()
                    )
                    replay = transformed["replay"]
                    data[i] = transformed["image"]
                    C, H, W = data[i].shape
                    new_data = torch.zeros((F, C, H, W))
                else:
                    transformed = A.ReplayCompose.replay(
                        replay, image=data[i], keypoints=traj[i].tolist()
                    )
                    data[i] = transformed["image"]

                new_data[i] = data[i]
                if len(transformed["keypoints"]) == 0:
                    print("failed transforming keypoints at idx: ", i)
                    print(
                        "transformed['keypoints']: ", transformed["keypoints"]
                    )
                    print("traj[i]: ", traj[i])
                else:
                    traj[i] = np.array(transformed["keypoints"])

            traj[:, :, 0] = np.where(
                traj[:, :, 0] < 0,
                np.ones_like(traj[:, :, 0]) * -1,
                traj[:, :, 0],
            )
            traj[:, :, 1] = np.where(
                traj[:, :, 1] < 0,
                np.ones_like(traj[:, :, 1]) * -1,
                traj[:, :, 1],
            )
            traj[:, :, 0] = np.where(
                traj[:, :, 0] > W,
                np.ones_like(traj[:, :, 0]) * -1,
                traj[:, :, 0],
            )
            traj[:, :, 1] = np.where(
                traj[:, :, 1] > H,
                np.ones_like(traj[:, :, 1]) * -1,
                traj[:, :, 1],
            )

            traj[:, :, 0] = np.where(
                traj[:, :, 1] < 0,
                np.ones_like(traj[:, :, 0]) * -1,
                traj[:, :, 0],
            )
            traj[:, :, 1] = np.where(
                traj[:, :, 0] < 0,
                np.ones_like(traj[:, :, 1]) * -1,
                traj[:, :, 1],
            )
            traj = torch.from_numpy(traj).long()
            # print("DEBUG: in kpdataloader, traj.shape: ", traj.shape)
            # new_data = new_data/256
            # new_data = torch.from_numpy(new_data).float()
            return (new_data, traj)
        else:
            return (data, traj)

    def read_full_vid(self, index):
        item_path = self.csv_data.loc[index, 0]  # Dir
        item_label = self.csv_data.loc[index, 1]  # Label
        # num_frames = len(glob.glob(item_path + "*.jpg"))
        num_frames = self.num_frames_array[index]
        target_idx = torch.from_numpy(np.array(int(item_label))).float()
        slice_object = slice(0, num_frames, 1)
        indices = np.arange(num_frames)[slice_object]
        frames = []
        for ind in indices:
            fpath = f"{item_path}{ind}.jpg"
            assert os.path.exists(fpath), fpath
            frames.append(cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB))

        traj_data = self.vid_traj_df_dic[item_path].copy()

        if len(frames) < (self.clip_size * self.nclips):
            frames.extend(
                [frames[-1]] * ((self.clip_size * self.nclips) - len(frames))
            )
        traj_data = self.get_traj_arrary(traj_data, indices, len(frames))
        data, traj_data = self._transform(frames, traj_data)

        return (data, target_idx, traj_data, item_path, frames, indices)

    def __getitem__(self, index, test_pos=-1):

        item_path = self.csv_data.loc[index, 0]  # Dir
        item_label = self.csv_data.loc[index, 1]  # Label
        # num_frames = len(glob.glob(item_path + "*.jpg"))
        num_frames = self.num_frames_array[index]
        # print('num_frames: ' + str(num_frames) + ' in vid: ' + str(item_path))
        assert num_frames > 0
        # print(num_frames)
        # target_idx = self.classes_dict[int(item.label)]
        target_idx = torch.from_numpy(np.array(int(item_label))).float()

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        # offset = 0
        offset = 1

        if test_pos >= 0:
            offset = test_pos * self.clip_size * self.step_size // 2
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
                # offset = torch.randint(0, diff, (1,)).item()  # add seed
                offset = torch.randint(1, diff, (1,)).item()  # start from 1 to diff-1
                # print("Kpdataloader, {} offset: {}".format(item_path, offset))

            slice_object = slice(
                offset,
                num_frames_necessary // self.nclips + offset,
                self.step_size,
            )

            clip_indices = np.arange(num_frames)[slice_object]

            #         frames, meta = self.gulp_directory[item.id, slice_object]
            for img_idx in clip_indices:
                try:
                    # for filenames are pure numbers
                    # fpath = f"{item_path}{img_idx}.jpg"  # Assuming no consecutive error imgs.

                    # for filenames are vid_name + frames
                    item_name = item_path.split('/')[-2] # get video name, i.e. vid_182c_trim

                    zidx = str(img_idx).zfill(4)    # pads string on the left with zeros, since frames between 0001 and 9999
                    img_name = item_name + '_' + str(zidx)
                    fpath = f"{item_path}{img_name}.png"
                    
                    # print(fpath)
                    if (
                        item_path in error_vid_frame.keys()
                        and (img_idx) in error_vid_frame[item_path]
                    ):
                        print("DEBUG: encountered error vid: {}".format(fpath))
                        correct_idx = img_idx-1 if img_idx>0 else img_idx+1
                        fpath = f"{item_path}{correct_idx}.jpg"  # Assuming no consecutive error imgs.
                        frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
                    else:
                        frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    print("DEBUG: catched encountered error vid: {}".format(fpath))
                    print(
                        "item_path in keys: {}, img_idx in value: {}".format(
                            (item_path in error_vid_frame.keys()),
                            (img_idx in error_vid_frame[item_path]),
                        )
                    )
                    correct_idx = img_idx-1 if img_idx>2 else img_idx+1
                    # get fpath
                    # fpath = f"{item_path}{correct_idx}.jpg"  # Assuming no consecutive error imgs.
                    item_name = item_path.split('/')[-2] # get video name, i.e. vid_182c_trim
                    zidx = str(img_idx).zfill(4)    # pads string on the left with zeros, since frames between 0001 and 9999
                    img_name = item_name + '_' + str(zidx)
                    fpath = f"{item_path}{img_name}.png"

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

        traj_data = self.vid_traj_df_dic[item_path].copy()

        traj_data = self.get_traj_arrary(traj_data, indices, len(frames))
        data, traj_data = self._transform(frames, traj_data)
        # data.shape= (batch_size, num_frames, encoder_dim, h, w)
        traj_map = self._traj_to_map(traj_data, data[0], 7, 0)
        if self.traj:
            traj_map_full = self._traj_to_map_gaussian(
                traj_data, data[0], data.shape[3], 6
            )

        return (
            data,
            target_idx,
            traj_map,
            item_path,
            frames,
            indices,
            traj_data,
            traj_map_full if self.traj else [],
        )

    def __len__(self):
        return len(self.csv_data)


def get_data_loaders(
    config,
    val_fold=4,
    test_fold=5,
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

    # transform = Compose([
    #                     ToPILImage(),
    #                     Resize(imsize),
    #                     CenterCrop(imsize),
    #                     ToTensor(),
    #                     # Normalize(mean=[0.485, 0.456, 0.406],
    #                     #           std=[0.229, 0.224, 0.225])
    #                     ])
    # video_transform_list = [
    #                         video_transforms.KPRandomRotation(30),
    #                         # video_transforms.KPRandomHorizontalFlip(),
    #                         # video_transforms.KPResize(imsize),
    #                         # video_transforms.KPRandomCrop(imsize),
    #                         volume_transforms.KPClipToTensor()
    #                         ]
    # kp_post_transform_list = [
    #                         video_transforms.Resize(8),
    #                         volume_transforms.ClipToTensor(),
    #                         ]
    # transform = video_transforms.Compose(video_transform_list)

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
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    # kp_transform = video_transforms.Compose(video_transform_list[:-1]+(kp_post_transform_list))

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, valid_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=val_fold,
        test_fold=test_fold,
        num_folds=num_folds,
    )

    train_loader = KeyPointDataset(
        input=train_df,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        is_val=False,
        transform=transform,
        traj=config.trajectory
        #  kp_transform = kp_transform
    )

    # val_video_transform_list = video_transform_list
    # val_video_transform_list = [
    #                         # video_transforms.Resize(imsize),
    #                         # video_transforms.CenterCrop(imsize),
    #                         volume_transforms.ClipToTensor()
    #                         ]
    # val_transform = video_transforms.Compose(val_video_transform_list)
    val_transform = A.ReplayCompose(
        [
            A.SmallestMaxSize(max_size=imsize),
            # A.Resize(height=imsize, width=imsize),
            A.CenterCrop(width=cropsize, height=cropsize),
            A.transforms.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    # val_transform = transform
    # val_kp_transform = video_transforms.Compose(val_video_transform_list[:-1]+(kp_post_transform_list))

    valid_loader = KeyPointDataset(
        input=valid_df,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        is_val=True,
        transform=val_transform,
        traj=config.trajectory
        #  kp_transform = val_kp_transform
    )

    test_loader = KeyPointDataset(
        input=test_df,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        is_val=True,
        transform=val_transform,
        traj=config.trajectory
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


class KPVideoFold(KeyPointDataset):
    def __init__(
        self,
        vid_idx,
        input,
        nclips,
        clip_size,
        step_size,
        is_val,
        transform=None,
    ):
        super(KPVideoFold, self).__init__(
            input, nclips, clip_size, step_size, is_val, transform
        )
        self.vid_idx = vid_idx

    def get_traj(self, traj, img_idx):
        ret_traj = np.zeros(12)
        if img_idx not in traj.index:
            ret_traj[:] = -1
        else:
            ret_traj[:] = traj.loc[img_idx].values
        ret_traj = np.reshape(ret_traj, (6, 2))
        return ret_traj

    def _transform(self, data, traj):
        data = [data]
        traj = np.reshape(traj, (1, *traj.shape))
        return super()._transform(data, traj)

    def __getitem__(self, img_idx, test_pos=-1):
        index = self.vid_idx
        item_path = self.csv_data.loc[index, 0]  # Dir
        item_label = self.csv_data.loc[index, 1]  # Label
        # num_frames = len(glob.glob(item_path + "*.jpg"))
        target_idx = torch.from_numpy(np.array(int(item_label))).float()
        fpath = f"{item_path}{img_idx}.jpg"
        assert os.path.exists(fpath), fpath
        try:
            if (
                item_path in error_vid_frame.keys()
                and (img_idx) in error_vid_frame[item_path]
            ):
                print("DEBUG: encountered error vid: {}".format(fpath))
                correct_idx = img_idx-1 if img_idx>0 else img_idx+1
                fpath = f"{item_path}{correct_idx}.jpg"  # Assuming no consecutive error imgs.
                frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print("DEBUG: catched encountered error vid: {}".format(fpath))
            print(
                "item_path in keys: {}, img_idx in value: {}".format(
                    (item_path in error_vid_frame.keys()),
                    (img_idx in error_vid_frame[item_path]),
                )
            )
            correct_idx = img_idx-1 if img_idx>0 else img_idx+1
            fpath = f"{item_path}{correct_idx}.jpg"  # Assuming no consecutive error imgs.
            frame = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)

        traj_data = self.vid_traj_df_dic[item_path].copy()

        traj_data = self.get_traj(traj_data, img_idx)
        # traj_data should have shape: (frame,6,2)
        # frame should have shape: (frame, C, H, W), data is a list of (C,H,W) np arrays
        data, traj_data = self._transform(frame, traj_data)
        traj_map = self._traj_to_map(traj_data, data[0], 7, 0)

        return (data, target_idx, traj_map, item_path, frame, traj_data)

    def __len__(self):
        return self.num_frames_array[self.vid_idx]


def get_all_img_data_loaders(
    config,
    fold=1,
    imsize=256,
    batch_size=512,
    clip_size=18,
    nclips=1,
    step_size=2,
    num_workers=0,
    dum=False,
    num_folds=5,
):
    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, val_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=fold,
        test_fold=fold,
        num_folds=num_folds,
    )
    dataloaders = []
    for i in range(len(test_df)):
        dataloaders.append(
            get_img_data_loader(
                config,
                i,
                fold,
                imsize,
                batch_size,
                clip_size,
                nclips,
                step_size,
                num_workers,
                dum,
                num_folds,
            )
        )

    return dataloaders


def get_img_data_loader(
    config,
    vid_idx,
    fold=1,
    imsize=256,
    batch_size=512,
    clip_size=18,
    nclips=1,
    step_size=2,
    num_workers=0,
    dum=False,
    num_folds=5,
):

    cropsize = int(imsize * 0.875)
    # cropsize = imsize

    fpath = folds_txt_path_dum if dum else folds_txt_path
    train_df, val_df, test_df = prepare_split(
        config=config,
        labels_path=fpath,
        val_fold=fold,
        test_fold=fold,
        num_folds=num_folds,
    )

    val_transform = A.ReplayCompose(
        [
            A.SmallestMaxSize(max_size=imsize),
            A.CenterCrop(width=cropsize, height=cropsize),
            A.transforms.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    test_loader = KPVideoFold(
        input=test_df,
        vid_idx=vid_idx,
        clip_size=clip_size,
        nclips=nclips,
        step_size=step_size,
        is_val=True,
        transform=val_transform,
        #  kp_transform = val_kp_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader

