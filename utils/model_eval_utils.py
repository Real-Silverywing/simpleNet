import os, sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pdb

from .cv2dataloader import prepare_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_preds(config, net, test_loader, test=False, folds=4, termout=False, traj=False):
    net.eval() 
    
    if not test:
        val_labels = torch.zeros(len(test_loader.dataset), 1)
    val_preds = torch.zeros(len(test_loader.dataset), config.num_classes).to(device)


    t0 = time.time()
    ll = len(test_loader)
    # no gradients during validation
    with torch.no_grad():
        for fold in range(folds):
            ci = 0
            for i, data in enumerate(test_loader):
                valid_imgs = data[0].float().to(device)
                if not test:
                    valid_labels = data[1].float().to(device).unsqueeze(1)

                # get predictions
                if config.model_name=="feattempnetwork":
                    seq_lengths = data[3]
                    label_vpreds = net(valid_imgs,seq_lengths)
                else:
                    # label_vpreds = net(valid_imgs, out_logits=(not config.weighted_bce))
                    label_vpreds = net(valid_imgs)
                if traj:
                    label_vpreds = label_vpreds[0]
                elif hasattr(config, "attention_loss") and config.attention_loss:
                    label_vpreds = label_vpreds[0]
                val_preds[ci: ci+label_vpreds.shape[0], :] += label_vpreds
                if not test:
                    val_labels[ci: ci+valid_labels.shape[0], :] = valid_labels
                ci = ci+label_vpreds.shape[0]

                if termout:
                    # make a cool terminal output
                    tc = time.time() - t0
                    tr = int(tc*(ll*folds - (ll*fold + i+1))/(ll*fold + i+1))
                    sys.stdout.write('\r')
                    sys.stdout.write('Fold: {}/{} | B: {:>3}/{:<3} | ETA: {:>4d}s'.
                        format(fold+1, folds, i+1, ll, tr))

        val_preds = val_preds.cpu()/folds

    print('')

    if test:
        return val_preds
    else:
        return val_preds, val_labels


def save_preds(config, net, test_loader, TEST_CSV_PATH, val_fold, test_fold, TTA_folds, 
                    best_th, gen_csv=True, termout=False, 
                    TK=True, traj=False, ky=False):
    print('Generating predictions...')

    SUBM_OUT = "./subm/{}/{}_val{}_test{}.csv".format(config.exp_name, config.model_name, 
str(val_fold), str(test_fold))
    print(SUBM_OUT)
    net.eval()

    if TK:
        print('Generating predictions...')
        test_preds = generate_TK_preds(net, test_loader, test=True, folds=TTA_folds, 
                                    termout=termout, traj=traj).numpy()
    else:
        print('Generating predictions...')
        test_preds = generate_preds(config, net, test_loader, test=True, folds=TTA_folds, 
                                    termout=termout, traj=traj).numpy()


    if gen_csv:
        if not ky:
            print('Saving predictions...')
            preds_df = pd.DataFrame(data=test_preds,columns=['Preds'])
            preds_df['th'] = pd.Series(best_th)
            preds_df.to_csv(SUBM_OUT.replace('subm', 'preds'), index=False) # save to ./preds




            _, _, test_df = prepare_split(config=config, val_fold=val_fold, test_fold=test_fold)
            test_df[1] = (test_preds>best_th).astype('int')
            test_df.to_csv(SUBM_OUT, header=False, index=False)
            print('Saved to ', SUBM_OUT)
        else:
            print('Saving predictions...')
            preds_df = pd.DataFrame(data=test_preds)
            preds_df['th'] = pd.Series(best_th)
            preds_df.to_csv(SUBM_OUT.replace('subm', 'preds'), index=False)

            # temp = pd.read_csv(TEST_CSV_PATH, header=0)
            # test_labels = temp['Label'].tolist()
            # temp = temp['Dir'].str.split(" ", n = 1, expand = True)
            # test_ids = temp[0].tolist()
            # test_df = pd.DataFrame({0: test_ids, 1: test_labels})
            test_df = test_loader.dataset.get_test_df()
            test_df[1] = (test_preds>best_th).astype('int')
            test_df.to_csv(SUBM_OUT, header=False, index=False)
            print('Saved to ', SUBM_OUT)

    return test_preds

def generate_TK_preds(net, test_loader, test=False, folds=5, termout=False, traj=False):
    net.eval() 
    
    if not test:
        val_labels = torch.zeros(len(test_loader.dataset), 1)
    val_preds = torch.zeros(len(test_loader.dataset), 1).to(device)

    t0 = time.time()
    ll = len(test_loader)
    # no gradients during validation
    ci = 0
    with torch.no_grad():
        test_dataset = test_loader.dataset
        for i in range(len(test_dataset)):
            test_pos = 0
            vid_preds = []
            while True:
                data = test_dataset.__getitem__(i, test_pos)
                if data is None:
                    break
                else:
                    valid_imgs = data[0].float().to(device)
                    pos_pred = net(valid_imgs.unsqueeze(0))
                    if traj:
                        pos_pred = pos_pred[0]
                    vid_preds.append(pos_pred)
                test_pos += 1
            
            label_vpreds = torch.tensor(vid_preds).mean(0, keepdim=True).unsqueeze(0).to(device)
            print(label_vpreds, label_vpreds.shape)
            val_preds[ci: ci+label_vpreds.shape[0], :] += label_vpreds
            ci = ci+label_vpreds.shape[0]

        val_preds = val_preds.cpu()

    print('')

    return val_preds

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "F1={:.3f}\nTh={:.3f}".format(ymax, xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data',textcoords="axes fraction",
                bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
