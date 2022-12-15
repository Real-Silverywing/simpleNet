import argparse
import os
import json
import pprint
from collections import namedtuple

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

    return config