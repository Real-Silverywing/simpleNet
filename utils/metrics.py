from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

def accuracy(preds, target):
    return np.mean((preds > 0.) == target)

def accuracy_general(preds, target):
    # preds.shape: (batch, c), should be numpy
    # target.shape: (batch,), should be numpy
    preds_index = np.argmax(preds,axis=1)
    return sum(preds_index==target)/target.shape[0]


def macro_f1(preds, target, class_wise=False):
    score = []
    for i in range(preds.shape[1]):
        score.append(f1_score(target[:, i], preds[:, i]))
    if class_wise:
        return np.array(score)
    else:
        return np.mean(score)

def f1_score_slow(y_true, y_pred, threshold=0.5):
    """
    Usage: f1_score(py_true, py_pred)
    """
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean((precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        print("Focal Loss with gamma = {}, alpha = {}, logits = {}".format(gamma, 
                    alpha, logits))
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        print("F1 Loss")

    def forward(self, input, target):
        tp = (target*input).sum(0)
        # tn = ((1-target)*(1-input)).sum(0)
        fp = ((1-target)*input).sum(0)
        fn = (target*(1-input)).sum(0)

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)

        f1 = 2*p*r / (p+r+1e-9)
        f1[f1!=f1] = 0.
        return 1 - f1.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Dice Loss")

    def forward(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2.*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalTverskyLoss(nn.Module):
    """
    https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    Focal Tversky Loss. Tversky loss is a special case with gamma = 1
    """
    def __init__(self, alpha = 0.4, gamma = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        print("Focal Tversky Loss with alpha = ", alpha, ", gamma = ", gamma)

    def tversky(self, input, target):
        smooth = 1.
        input = torch.sigmoid(input)

        target_pos = target.view(-1)
        input_pos = input.view(-1)
        true_pos = (target_pos * input_pos).sum()
        false_neg = (target_pos * (1-input_pos)).sum()
        false_pos = ((1-target_pos)*input_pos).sum()
        return (true_pos + smooth)/(true_pos + self.alpha*false_neg + \
                        (1-self.alpha)*false_pos + smooth)

    def forward(self, input, target):
        pt_1 = self.tversky(input, target)
        return (1-pt_1).pow(self.gamma)

class DiceLoss(nn.Module):
    def __init__(self, p=2, smooth=1, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predict, target):
        # predict.shape: (batch, frames, L)
        # target.shape: (batch, frames, L)
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class SpatialAttentionLoss(nn.Module):

    def __init__(self, ori_size=224, loss_type="dice"):
        super(SpatialAttentionLoss, self).__init__()
        print("Spatial Attention Loss")
        self.ori_size=ori_size
        self.dice_loss = DiceLoss()
        self.mse = nn.MSELoss()
        self.loss_type = loss_type

    def _gather(self, x, idx):
        idx = idx.contiguous()
        idx_shape = idx.shape
        idx = idx.view(-1, idx_shape[-2], idx_shape[-1]) # should be [frames*batchsize, 6, 2]
        unvalid_idx = idx[:,:,0]<0
        valid_idx = idx[:,:,0]>=0
        x = x.contiguous()
        x_shape = x.shape
        L = x_shape[-1]**0.5
        down_scale = self.ori_size//L
        idx = idx//down_scale
        lin_idx = idx[:,:,1] + L*(idx[:,:,0]) # should be [frames*batchsize, 6]
        lin_idx = lin_idx.long()
        # print("DEBUG: lin_idx.max: \t lin_idx.min: ",torch.max(lin_idx), torch.min(lin_idx))
        lin_idx[unvalid_idx]=0
        x = x.view(-1, x.size(-1)) # should be [frames*batchsize, 64]
        # print("DEBUG: x.shape: \t lin_idx.shape: ",x.shape, lin_idx.shape)
        ret = x.gather(-1, lin_idx)
        ret[unvalid_idx]=1
        return ret, valid_idx
    
    def focus_only(self, keypoints, attention):
        """
            keypoints: (batch, frames, 6, 2)
            attention: (batch, frames, batch, HW)
        """
        attention = F.softmax(attention, dim=2)
        # F,B,HW = attention.shape
        # L = HW**0.5
        # L = self.ori_size//L
        gathered, valid = self._gather(attention, keypoints)
        # loss = torch.mean(1-self._gather(attention,keypoints))
        loss = torch.sum(1-gathered)/max(1,torch.sum(valid))
        if loss is None:
            print("DEBUG: error, attention loss is None, gathered: ", gathered)
            print("DEBUG: valid: ", valid)
            print("DEBUG: sum(valid)", sum(valid))
            print("DEBUG: torch.sum(1-gathered): ", torch.sum(1-gathered))
        return loss
    
    def focus_only_map_dice(self, traj_map, attention):
        """
            traj_map: (batch, frames, 7, 7) 
            attention: (batch, frames, HW), value range: 0-1
        """
        attention = torch.sigmoid(attention)
        B,Fr,L,_ = traj_map.shape
        traj_map = traj_map.contiguous().view(B,Fr,L*L)
        return self.dice_loss(attention, traj_map)

    def focus_only_map_mse(self, traj_map, attention):
        """
            traj_map: (batch, frames, 7, 7) or (batch,frames, 224, 224)
            attention: (batch, frames, HW), value range: 0-1
        """
        attention = torch.sigmoid(attention)
        B,Fr,L,_ = traj_map.shape
        traj_map = traj_map.contiguous().view(B,Fr,L*L)
        return self.mse(attention, traj_map)
    
    def forward(self, keypoints, attention):
        """
            now presume batchsize=1
        """
        attention = attention.contiguous().permute(1,0,2)
        # print("DEBUG: attention.shape: ", attention.shape)
        # print("DEBUG: keypoints.shape: , dtype: ", keypoints.shape, keypoints.dtype)
        # return self.focus_only_map_dice(keypoints,attention)
        if self.loss_type=="dice":
            return self.focus_only_map_dice(keypoints,attention)
        else:
            return self.focus_only_map_mse(keypoints, attention)