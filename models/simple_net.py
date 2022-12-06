import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet101, resnet18, resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0, attnvis = False):
        super(SimpleNet, self).__init__()

        self.rgb_base = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2]) # resnet 18
        # self.rgb_base = nn.Sequential(*list(resnet101(pretrained=True).children())[:-2]) # resnet 101


        for p in self.rgb_base.parameters():
            p.requires_grad = False

        # for p in self.rgb_base[-1].parameters():
        #     p.requires_grad = True


        # Pooling

        self.low_avgpool = nn.AvgPool2d(7)
        # self.low_maxpool = nn.MaxPool2d(7)
        # self.relu_maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)

        # Temporal aggregation
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True) # resnet 18
        # self.lstm = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True) # resnet 101
        # self.lstm_paper = nn.LSTM(input_size=4608, hidden_size=256, batch_first=True) # paper

        self.last_linear = nn.Linear(256, 1, bias=True)
        # self.last_linear = nn.Linear(256, 1, bias=True) # paper


    def forward(self, X):

        X = X[0,...].permute(1, 0, 2, 3)    # frames x 3 x h x w

        X = self.rgb_base(X) # frames x hidden_size x 7 x 7

        X = self.low_avgpool(X)     # frames x hidden_size x 1 x 1
        
        X = X.squeeze().unsqueeze(0)    # 1 x frames x LSTM_input (bs x seq_len x embed_dim)

        X, hidden = self.lstm(X, None) # X: 1 x frames x hidden_size   

        X = self.last_linear(X)  #1 x frames x 1   

        X = torch.sigmoid(X)

        X = X[:, -1, :] # 1 x last hidden x 1

        # X = X[0,...]    # frames x 3 x h x w
        # X = self.rgb_base(X) # frames x hidden_size x 7 x 7
        # X = self.relu_maxpool(X) # frames x hidden_size x 3 x 3 (according to the paper):256*512*3*3
        # X = X.reshape((256,-1)).unsqueeze(0)    # 1*frames*4608: 4608=512*3*3
        # X, hidden = self.lstm_paper(X, None)
        # X = self.last_linear(X)
        # X = torch.sigmoid(X)
        # X = X[:, -1, :] # 1 x last hidden x 1
        

        return X

