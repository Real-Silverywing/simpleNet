import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0, attnvis = False):
        super(SimpleNet, self).__init__()

        self.rgb_base = nn.Sequential(*list(resnet101(pretrained=True).children())[:-2])
        for p in self.rgb_base.parameters():
            p.requires_grad = False

        # for p in self.rgb_base[-1].parameters():
        #     p.requires_grad = True

        # Pooling
        # self.low_avgpool = nn.AvgPool2d(8)
        # self.low_maxpool = nn.MaxPool2d(8)
        self.low_avgpool = nn.AvgPool2d(7)
        self.low_maxpool = nn.MaxPool2d(7)

        # Temporal aggregation
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, batch_first=True)

        self.last_linear = nn.Linear(256, 1, bias=True)


    def forward(self, X):
        # X = X[0,...].permute(1, 0, 2, 3) 
        X = X[0,...]    # framesx3xhxw
        X = self.rgb_base(X) # framesx2048x8x8

        X = self.low_avgpool(X)     # framesx2048x1x1
        X = X.squeeze().unsqueeze(0)    # 1xframesx2048 (bs x seq_len x embed_dim)

        X, hidden = self.lstm(X, None)

        X = self.last_linear(X)

        X = torch.sigmoid(X)
        X = X[:, -1, :]
        # print(X.shape, X)

        return X