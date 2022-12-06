# 1. run data/extract_frames.py
# 2. run data/prepare_test_set.py
# 3. python new_test.py --TK


import os, sys
import time
import argparse
import json
import pprint
from collections import namedtuple
from itertools import permutations
import cv2
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import gc
from tqdm import tqdm

import models.model_list as model_list
from utils.cv2dataloader import get_data_loaders as get_cv2dataloaders
from utils.kpdataloader import get_data_loaders as get_kpdataloaders
from utils.segment_dataloader import get_data_loaders as get_segment_dataloaders
from utils.misc import log_metrics, cosine_annealing_lr, process_config, set_random_seed
from utils.dataloader_utils import get_dataloaders_from_config
from utils.metrics import accuracy, FocalLoss, SpatialAttentionLoss

from utils.model_eval_utils import generate_preds, save_preds
from utils.torch_videovision.videotransforms import video_transforms, volume_transforms


parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', type = str,
                    help="run configuration")
parser.add_argument('--dum', action='store_true', default=False,
                    help='train on the dum folds')
parser.add_argument('--TK', action='store_true', default=False,
                    help='Evaluate TK style')
parser.add_argument('--latest', action='store_true', default=False,
                    help='Load the latest checkpoint (default best)')
parser.add_argument('-p', '--print', action='store_true', default=False,
                    help='Print real time status')
parser.add_argument('--split', type=int, choices=range(-1, 20), default=-1, 
                    help='Which split to train on (0-19)')
parser.add_argument("--num_folds", default=5, help="Total number of folds")
args = parser.parse_args()

config = process_config(args.config)

TTA_folds = config.TTA_folds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using CUDA")
    cudnn.benchmark = True

if not os.path.exists('./model_weights/'+config.exp_name):
    os.makedirs('./model_weights/'+config.exp_name)
if not os.path.exists('./logs/'+config.exp_name):
    os.makedirs('./logs/'+config.exp_name) 
if not os.path.exists('./preds/'+config.exp_name):
    os.makedirs('./preds/'+config.exp_name) 
if not os.path.exists('./subm/'+config.exp_name):
    os.makedirs('./subm/'+config.exp_name) 

test_csv_path = "./data/new_all_labels.csv"  # Made using prepare_test_set.py

# resume training
def load_checkpoint(model, model_ckpt):
    # Note: Input model should be pre-defined.  This routine only updates its state.

    checkpoint = torch.load(model_ckpt, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint '{}' (epoch {})"
              .format(model_ckpt, checkpoint['epoch']))

    model = model.to(device)
    model.eval()
    return model

def test_network(config, net, model_ckpt, fold=0, val_fold=4, test_fold=5, best_th=0.5, TEST_CSV_PATH = 
test_csv_path):
    # test the network

    # get the loaders
    train_loader, valid_loader, test_loader = get_dataloaders_from_config(config, val_fold, test_fold, args.dum, args.num_folds)
    
    # test_df = pd.read_csv(TEST_CSV_PATH)

    if args.latest:
        model_ckpt = model_ckpt.replace('best', 'latest')

    try:
        net = load_checkpoint(net, model_ckpt)
    except FileNotFoundError:
        print("=> no checkpoint found at '{}'".format(model_ckpt))
        return

   # subm_out = "./subm/{}/new_{}_val{}_test{}.csv".format(config.exp_name, 
    #                config.model_name, str(val_fold), str(test_fold))
        
    save_preds(config, net, test_loader, TEST_CSV_PATH, val_fold, test_fold, 
                    TTA_folds, best_th,
                    termout=args.print, TK=args.TK, traj=config.trajectory)

def main_test(config, val_fold, test_fold):

    set_random_seed()

    model_params = [config.exp_name, config.model_name, str(val_fold), str(test_fold)]
    MODEL_CKPT = './model_weights/{}/best_{}_val{}_test{}.pth'.format(*model_params)

    # net = Atlas_DenseNet(model = config.model_name, bn_size=4, drop_rate=config.drop_rate)
    Net = getattr(model_list, config.model_name)
    
    net = Net(config, drop_rate=config.drop_rate)

    net = nn.parallel.DataParallel(net)
    net.to(device)
    VAL_FOLD = val_fold
    TEST_FOLD = test_fold
    test_network(config, net, model_ckpt=MODEL_CKPT, val_fold=VAL_FOLD, test_fold=TEST_FOLD)

if __name__ == '__main__':
    inval_fold, intest_fold = list(permutations(range(1,6), 2))[args.split]
    print("*************************************")
    print("Testing model with Val Fold = {}, Test Fold = {}".format(inval_fold, intest_fold))

    main_test(config, val_fold=inval_fold, test_fold=intest_fold)
    print('')