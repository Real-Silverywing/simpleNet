import os, sys
import time
import argparse
import json
import shutil
import pprint
from collections import namedtuple
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision

import gc
from tqdm import tqdm

import models.model_list as model_list
from utils.cv2dataloader import get_data_loaders
from utils.preprocessed_dataloader import (
    get_data_loaders as get_processed_data_loaders,
)
from utils.segment_dataloader import get_data_loaders as get_segment_dataloaders
from utils.kpdataloader import get_data_loaders as get_kpdataloaders
from utils.feature_dataloader import get_data_loaders as get_featdataloaders
from utils.misc import log_metrics, cosine_annealing_lr, process_config, set_random_seed
from utils.metrics import accuracy, FocalLoss, SpatialAttentionLoss
from utils.dataloader_utils import get_dataloaders_from_config

from utils.model_eval_utils import generate_preds, save_preds


parser = argparse.ArgumentParser(description="Atlas Protein")
parser.add_argument(
    "--config", default="/projects/colonoscopy/group_scratch/xzhou86/simpleNet/configs/test_config.json", help="run configuration"
)
parser.add_argument(
    "--dev_mode",
    action="store_true",
    default=False,
    help="train only few batches per epoch",
)
parser.add_argument(
    "--dum", action="store_true", default=False, help="train on the dum folds"
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from the checkpoint",
)
parser.add_argument(
    "--latest",
    action="store_true",
    default=False,
    help="Load the latest checkpoint (default best)",
)
parser.add_argument(
    "--tboard", action="store_true", default=False, help="Log on tensorboard"
)
parser.add_argument(
    "--subm",
    action="store_true",
    default=False,
    help="Generate the test predictions submissions",
)
parser.add_argument(
    "--split",
    type=int,
    choices=range(-1, 20),
    default=-1,
    help="Which split to train on (0-19)",
)
parser.add_argument(
    "-p",
    "--print",
    action="store_true",
    default=False,
    help="Print real time status",
)
parser.add_argument("--num_folds", default=4, help="Total number of folds")
args = parser.parse_args()

config = process_config(args.config)

# if not os.path.exists('./configs/old_configs/'+config.exp_name):
#    os.makedirs('./configs/old_configs/'+config.exp_name)
#    shutil.copy2('./configs/config.json', './configs/old_configs/{}/config.json'
#                    .format(config.exp_name))

num_folds = args.num_folds


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using CUDA")
    cudnn.benchmark = True
else:
    print("**********************************")
    print("********* NOT USING CUDA *********")
    print("**********************************")

if not os.path.exists("./model_weights/" + config.exp_name):
    os.makedirs("./model_weights/" + config.exp_name)
if not os.path.exists("./logs/" + config.exp_name + "/board"):
    os.makedirs("./logs/" + config.exp_name + "/board")
if not os.path.exists("./preds/" + config.exp_name):
    os.makedirs("./preds/" + config.exp_name)
if not os.path.exists("./subm/" + config.exp_name):
    os.makedirs("./subm/" + config.exp_name)

if args.tboard:
    from utils.logger import Logger as TfLogger

    # from tensorboardX import SummaryWriter
    logger = TfLogger("./logs/" + config.exp_name + "/board")
    # graph_written = False

# resume training
def load_checkpoint(model, optimizer, model_ckpt):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0

    if args.latest:
        model_ckpt = model_ckpt.replace("best", "latest")

    print("Resuming from checkpoint '{}'".format(model_ckpt))

    if os.path.isfile(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val_loss = checkpoint["best_val_loss"]
        print(
            "Loaded checkpoint '{}' (epoch {})".format(
                model_ckpt, checkpoint["epoch"]
            )
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


# training function
def train(net, optimizer, loss, train_loader, freeze_bn=False):
    """
    uses the data loader to grab a batch of images
    pushes images through network and gathers predictions
    updates network weights by evaluating the loss functions
    """
    # set network to train model
    net.train()

    # keep track of our loss
    iter_loss = 0.0
    iter_attn_loss = 0.0

    t0 = time.time()
    ll = len(train_loader)
    # loop over the images for the desired amount
    print("length train_loader: " + str(ll))
    retain_graph = True
    count = 0

    # calculate weight

    # lb = []
    # for data in train_loader:
    #     lb.append(data[1].item())
    # c1 = lb.count(1)
    # c0 = lb.count(0.0)
    class_weights = get_weights_inverse_num_of_samples(2, [8, 30], power = 1)



    for i, data in enumerate(train_loader):
        # data = next(iter(train_loader))       # Never do this

        imgs = data[0].to(device)
        labels = data[1].to(device)
        if hasattr(config, "attention_loss") and config.attention_loss:
            attn_labels = data[2].to(device)
        if config.trajectory:
            traj_labels = data[7].to(device)

        if config.fp16:
            net = net.half()
            imgs = imgs.half()
            labels = labels.half()

        # get predictions
        if config.trajectory:
            label_preds, traj_preds = net(imgs) # traj_preds has shape (Batch,Frames,49,6)
            traj_preds = torch.sum(traj_preds,dim=3)
        elif hasattr(config, "attention_loss") and config.attention_loss:
            label_preds, attentions = net(imgs)
        
        elif config.model_name=="feattempnetwork":
            seq_lengths = data[3]
            label_preds = net(imgs,seq_lengths)
        else:
            label_preds = net(imgs)
        # print(label_preds, labels)
        # calculate loss
        if config.fp16:
            label_preds, labels = label_preds.float(), labels.float()

        # print(label_preds.shape, labels.shape)
        # print("Preds = ", label_preds.data, "Label = ", labels.data)
        if config.trajectory:
            class_loss = loss[0](label_preds, labels.unsqueeze(1))
            # traj_labels should have shape (Batch, frames, 224,224,6)
            traj_labels = torch.sum(traj_labels,dim=4)
            traj_loss = loss[1](traj_labels, traj_preds)
            tloss = class_loss + config.attention_lambda*traj_loss

        elif hasattr(config, "attention_loss") and config.attention_loss:
            class_loss = loss[0](label_preds, labels.unsqueeze(1))
            attn_loss = loss[1](attn_labels, attentions)
            tloss = class_loss + config.attention_lambda * attn_loss
            # tloss = attn_loss
        else:
            if config.num_classes == 1:
                if config.weighted_bce:
                    # weightd loss
                    if labels.item() == 0:
                        tloss = class_weights[0] * loss(label_preds, labels.unsqueeze(1))
                    elif labels.item() == 1:
                        tloss = class_weights[1] * loss(label_preds, labels.unsqueeze(1))


                else:
                    # one-class without weight
                    tloss = loss(label_preds, labels.unsqueeze(1))
            else:
                # multi class
                labels_one_hot = torch.zeros_like(label_preds)
                for i in range(labels.shape[0]):
                    labels_one_hot[i, labels[i].long()] = 1
                tloss = loss(label_preds, labels_one_hot)

        # coin = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             # print(type(obj), obj.size())
        #             coin += 1
        #     except:
        #         pass
        # print(coin)

        # zero gradients from previous run
        optimizer.zero_grad()
        # calculate gradients
        if config.fp16:
            optimizer.backward(tloss)
        else:
            tloss.backward(retain_graph=False)

        # update weights
        optimizer.step()

        # get training stats
        iter_loss += float(tloss.item())
        if hasattr(config, "attention_loss") and config.attention_loss:
            iter_attn_loss += float(attn_loss.item())
        elif config.trajectory:
            iter_attn_loss += float(traj_loss.item())

        # Log graphsummary
        # global graph_written
        # if not graph_written:
        #     net.eval()
        #     with SummaryWriter('./logs/'+config.exp_name+'/board') as w:
        #         w.add_graph(net, imgs, True)
        #     net.train()
        #     graph_written = True

        if args.print:
            # make a cool terminal output
            tc = time.time() - t0
            tr = int(tc * (ll - i - 1) / (i + 1))

            sys.stdout.write("\r")
            sys.stdout.write(
                "B: {:>3}/{:<3} | Loss: {:<7.4f} | ETA: {:>4d}s".format(
                    i + 1, ll, tloss.item(), tr
                )
            )

        if i == 5 and args.dev_mode == True:
            print("\nDev mode on. Prematurely stopping epoch training.")
            break

    epoch_loss = iter_loss / (len(train_loader.dataset) / config.batch_size)
    epoch_attn_loss = iter_attn_loss / (
        len(train_loader.dataset) / config.batch_size
    )
    if hasattr(config, "attention_loss") and config.attention_loss:
        print(
            "\n"
            + "Avg Train Loss: {:.4}\t Avg Class Loss: {:.4}\t Avg Attn Loss: {:.4}\t Avg Attn Loss absolut: {:.4}".format(
                epoch_loss,
                epoch_loss - config.attention_lambda * epoch_attn_loss,
                config.attention_lambda * epoch_attn_loss,
                epoch_attn_loss,
            )
        )
    elif config.trajectory:
        print(
            "\n"
            + "Avg Train Loss: {:.4}\t Avg Class Loss: {:.4}\t Avg Traj Loss: {:.4}\t Avg Traj Loss absolut: {:.4}".format(
                epoch_loss,
                epoch_loss - config.attention_lambda * epoch_attn_loss,
                config.attention_lambda * epoch_attn_loss,
                epoch_attn_loss,
            )
        )
    else:
        print("\n" + "Avg Train Loss: {:.4}".format(epoch_loss))

    return epoch_loss


# validation function
def valid(net, optimizer, loss, valid_loader, save_imgs=False, fold_num=0):
    net.eval()
    # keep track of preds
    if config.fp16:
        net = net.float()

    val_preds, val_labels = generate_preds(
        config,
        net,
        valid_loader,
        folds=config.TTA_folds,
        termout=args.print,
        traj=config.trajectory,
    )

    if config.trajectory:
        epoch_vloss = loss[0](val_preds, val_labels)
    elif hasattr(config, "attention_loss") and config.attention_loss:
        epoch_vloss = loss[0](val_preds, val_labels)
    else:
        if config.num_classes == 1:
            epoch_vloss = loss(val_preds, val_labels)
        else:
            labels_one_hot = torch.zeros_like(val_preds)
            for i in range(val_preds.shape[0]):
                labels_one_hot[i, val_labels[i].long()] = 1
            epoch_vloss = loss(val_preds, labels_one_hot)

    print("preds    lables")
    print(val_preds, val_labels)
    if config.num_classes == 1:
        epoch_vacc = accuracy(val_preds.numpy() > 0.5, val_labels.numpy())
    else:
        epoch_vacc = (
            val_preds.argmax(1) == val_labels[:, 0]
        ).sum().float() / val_labels.shape[0]
    print(
        "Avg Eval Loss: {:.4}, Avg Eval Acc. {:.4}".format(
            epoch_vloss, epoch_vacc
        )
    )
    return epoch_vloss, epoch_vacc


def train_network(net, model_ckpt, fold=0, val_fold=4, test_fold=5):
    # train the network, allow for keyboard interrupt

    if config.fp16:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config.lr,
            eps=1e-04,
        )

        from apex.fp16_utils import FP16_Optimizer

        optimizer = FP16_Optimizer(
            optimizer, dynamic_loss_scale=True, verbose=False
        )

    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr
        )

    valid_patience = 0
    best_val_loss = None
    best_val_acc = None
    cycle = 0
    t_ = 0

    if args.resume and os.path.isfile(model_ckpt):
        net, optimizer, start_epoch, best_val_loss = load_checkpoint(
            net, optimizer, model_ckpt
        )

    if config.reduce_lr_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", config.lr_scale, config.lr_patience, True
        )

    if config.cosine_annealing:
        cos_lr, cycle_ends = cosine_annealing_lr(
            config.min_lr,
            config.max_lr,
            config.cycle_size,
            config.epochs,
            config.cycle_size_inc,
        )

    # get the loaders
    train_loader, valid_loader, test_loader = get_dataloaders_from_config(config, val_fold, test_fold, args.dum, args.num_folds)

    # loss = F1Loss()
    # if hasattr(config, 'focal_gamma'):
    #     loss = FocalLoss(gamma=config.focal_gamma, logits=False)
    # else:
    #     loss = FocalLoss(logits=False)
    if config.trajectory:
        loss = (nn.BCELoss().cuda(), SpatialAttentionLoss(config.imsize,"mse").cuda())
    elif hasattr(config, "attention_loss") and config.attention_loss:
        loss = (nn.BCELoss().cuda(), SpatialAttentionLoss(config.imsize).cuda())
    else:
        loss = nn.BCELoss().cuda()

    # loss = nn.BCEWithLogitsLoss().cuda()
    # if hasattr(config, 'focal_gamma'):
    #     loss = FocalTverskyLoss(gamma = config.focal_gamma)
    # else:
    #     loss = FocalTverskyLoss()

    # training flags
    freeze_bn = False
    save_imgs = False
    train_losses = []
    valid_losses = []
    valid_accs = []
    lr_hist = []

    print("Training ...")
    print("Saving to ", model_ckpt)
    for e in range(config.epochs):
        print("\n" + "Epoch {}/{}".format(e, config.epochs))

        start = time.time()

        t_l = train(net, optimizer, loss, train_loader, freeze_bn)

        v_l, v_acc = valid(net, optimizer, loss, valid_loader, save_imgs, fold)

        if config.reduce_lr_plateau:
            scheduler.step(v_l)

        if config.cosine_annealing:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cos_lr[e]
            if e in cycle_ends:
                cycle = np.where(cycle_ends == e)[0][0] + 1
                net.eval()
                torch.save(
                    net.state_dict(),
                    model_ckpt.replace("best", "cycle{}".format(cycle)),
                )
                print(
                    "Cycle {} completed. Saving model to {}".format(
                        cycle,
                        model_ckpt.replace("best", "cycle{}".format(cycle)),
                    )
                )

        lr_hist.append(optimizer.param_groups[0]["lr"])

        state = {
            "epoch": e,
            "arch": config.model_name,
            "state_dict": net.state_dict(),
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(),
        }

        # save the model on best validation loss
        if best_val_loss is None or v_l < best_val_loss:
            best_val_loss = v_l
            state["best_val_loss"] = best_val_loss

            torch.save(state, model_ckpt)
            valid_patience = 0
            print(
                "Best val loss achieved. loss = {:.4f}.".format(v_l),
                " Saving model to ",
                model_ckpt,
            )

        # save the model on best validation f1
        # if best_val_acc is None or v_acc > best_val_acc:
        #     net.eval()
        #     torch.save(net.state_dict(), model_ckpt.replace('best', 'bestf1'))
        #     best_val_acc = v_acc
        #     valid_patience = 0
        #     print('Best val F1 achieved. F1 = {:.4f}.'.
        #         format(v_acc), " Saving model to ", model_ckpt.replace('best', 'bestf1'))

        # if (e > 5):
        #     SUBM_OUT = './subm/{}_{}_epoch{}.csv'.format(
        #                     config.model_name, config.exp_name, str(e))
        #     save_preds(net, config, SUBM_OUT)

        else:
            valid_patience += 1
            if valid_patience >= config.es_patience:
                print("Early stopping patience reached, stopping...")
                break

        torch.save(state, model_ckpt.replace("best", "latest"))

        train_losses.append(t_l)
        valid_losses.append(v_l)
        valid_accs.append(v_acc)

        log_metrics(
            train_losses,
            valid_losses,
            valid_accs,
            lr_hist,
            e,
            model_ckpt,
            config,
        )

        if args.tboard:
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = {
                "train_loss": t_l,
                "val_loss": v_l.item(),
                "val_accuracy": v_acc,
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, e + 1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in net.named_parameters():
                tag = tag.replace(".", "/")
                logger.histo_summary(tag, value.data.cpu().numpy(), e + 1)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + "/grad", value.grad.data.cpu().numpy(), e + 1
                    )

            # 3. Log training images (image summary)
            # info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

            # for tag, images in info.items():
            #     logger.image_summary(tag, images, step+1)

        t_ += 1
        print("Time: {:d}s".format(int(time.time() - start)))

    if args.subm:
        SUBM_OUT = "./subm/{}/{}_val{}_test{}.csv".format(
            config.exp_name, config.model_name, str(val_fold), str(test_fold)
        )
        save_preds(
            net,
            test_loader,
            val_fold=val_fold,
            test_fold=test_fold,
            TTA_folds=1,
            best_th=0.5,
            SUBM_OUT=SUBM_OUT,
            gen_csv=True,
            termout=args.print,
        )

def get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, power = 1):
    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def get_weights_effective_num_of_samples(no_of_classes, beta, samples_per_cls):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights_for_samples = (1.0 - beta) / np.array(effective_num)
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def main_train(val_fold=num_folds + 1, test_fold=num_folds):

    model_params = [
        config.exp_name,
        config.model_name,
        str(val_fold),
        str(test_fold),
    ]
    MODEL_CKPT = "./model_weights/{}/best_{}_val{}_test{}.pth".format(
        *model_params
    )

    # net = Atlas_DenseNet(model = config.model_name, bn_size=4, drop_rate=config.drop_rate)
    Net = getattr(model_list, config.model_name)

    net = Net(config, drop_rate=config.drop_rate)

    net = nn.parallel.DataParallel(net)
    net.to(device)

    train_network(
        net, model_ckpt=MODEL_CKPT, val_fold=val_fold, test_fold=test_fold
    )


if __name__ == "__main__":
    set_random_seed()
    if args.split == -1:
        kill_count = 0
        kill_time = time.time()
        for val_fold, test_fold in permutations(range(1, num_folds + 1), 2):
            try:
                print("*************************************")
                print(
                    "Training model with Val Fold = {}, Test Fold = {}".format(
                        val_fold, test_fold
                    )
                )
                main_train(val_fold, test_fold)
                print("")

            except KeyboardInterrupt:
                if time.time() - kill_time < 5:
                    kill_count += 1
                else:
                    kill_count = 0

                kill_time = time.time()
                if kill_count > 3:
                    print("Exiting")
                    break

    else:
        val_fold, test_fold = list(permutations(range(1, num_folds + 1), 2))[
            args.split
        ]
        print("*************************************")
        print(
            "Training model with Val Fold = {}, Test Fold = {}".format(
                val_fold, test_fold
            )
        )
        main_train(val_fold, test_fold)
        print("")

