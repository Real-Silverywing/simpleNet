import argparse
import os
import json
import pprint

import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple

def set_random_seed(seed=23):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # forbid hash random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_metrics(train_losses, valid_losses, valid_accs, lr_hist, e, model_ckpt, config):
    mpl.use('Agg')
    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].plot(train_losses)
    axes[0, 1].plot(valid_losses)
    axes[1, 0].plot(valid_accs)
    axes[1, 1].plot(lr_hist)
    axes[0, 0].set_title('Train Loss')
    axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_title('Val Acc')
    axes[1, 1].set_title('LR History')
    plt.suptitle("At Epoch {}, desc: {}".format(e+1, config.desc), fontsize=16)
    plt.savefig(model_ckpt.replace('model_weights', 'logs').replace('.pth', '.png'))
    plt.close('all')

def cosine_annealing_lr(min_lr, max_lr, cycle_size, epochs, cycle_size_inc = 0):
    new_epochs = cycle_size
    n_cycles = 1
    temp_cs = cycle_size
    while (new_epochs <= epochs-temp_cs):
        temp_cs += cycle_size_inc
        new_epochs += temp_cs
        n_cycles += 1
    print("Performing {} epochs for {} cycles".format(new_epochs, n_cycles))
    
    cycle_e = 0
    lr = []
    cycle_ends = [0]
    for e in range(new_epochs):
        lr.append(min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(cycle_e*np.pi/cycle_size)))
        cycle_e += 1
        if cycle_e == cycle_size:
            cycle_ends.append(cycle_e + cycle_ends[-1])
            cycle_e = 0
            cycle_size += cycle_size_inc
    cycle_ends = np.array(cycle_ends[1:]) - 1
    return lr, cycle_ends

def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                        img.astype("uint8"))

def process_config(config_path):
    with open(config_path) as f_in:
        d = json.load(f_in)
        if "dataloader" not in d.keys():
            d["dataloader"] = "kpdataloader"
        if "use_rgb_base" not in d.keys():
            d["use_rgb_base"] = True
        if "select" not in d.keys():
            d["select"] = False
        if "decoder_dim" not in d.keys():
            d["decoder_dim"] = 256
        if "model_type" not in d.keys():
            d["model_type"] = ""
        if "processed_run" not in d.keys():
            d["processed_run"] = ""
        if "temporal_attention" not in d.keys():
            d["temporal_attention"] = True
        if "force_subsample" not in d.keys():
            d["force_subsample"] = False
        if "temporal_attention_softmax" not in d.keys():
            d["temporal_attention_softmax"] = True
        if "spatial_attn" not in d.keys():
            d["spatial_attn"] = True
        if "spatial_attn_pool" not in d.keys():
            d["spatial_attn_pool"] = False
        if "unfreeze_rgb" not in d.keys():
            d["unfreeze_rgb"] = False
        if "group_drsl" not in d.keys():
            d["group_drsl"] = False
        if "exclude_ext_data" not in d.keys():
            d["exclude_ext_data"] = False
        if "weighted_bce" not in d.keys():
            d["weighted_bce"] = False
        if "pos_weight" not in d.keys():
            d["pos_weight"] = 1
        if "random_interval" not in d.keys():
            d["random_interval"] = False
        if "group_label" not in d.keys():
            d["group_label"] = False
        if "decoder" not in d.keys():
            d["decoder"] = "LSTM"
        if "disc_exp_name" not in d.keys():
            d["disc_exp_name"] = ""

        config = namedtuple("config", d.keys())(*d.values())
    print("Loaded configuration from ", config_path)
    print("")
    pprint.pprint(d)
    # time.sleep(5)
    print("")
    return config
