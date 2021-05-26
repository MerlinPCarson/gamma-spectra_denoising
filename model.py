import torch
import torch.nn as nn
import math


def init_weights(model):
    for layer in model:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant_(layer.bias.data, 0.0)


class DnCNN(nn.Module):
    def __init__(self, num_channels=1, num_layers=17, kernel_size=3, stride=1, num_filters=64, dilation_rate=3):
        super(DnCNN, self).__init__()

        padding = int((kernel_size-1)/2)
        padding_d = math.floor((kernel_size + (kernel_size-1) * (dilation_rate-1))/2)

        # create module list
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_filters, stride=stride, kernel_size=kernel_size, padding=padding, bias=False))
        self.layers.append(nn.LeakyReLU())
        #self.layers.append(nn.AvgPool1d(2))
        for i in range(num_layers-2):
            self.layers.append(nn.Conv1d(in_channels=num_filters, dilation=dilation_rate, out_channels=num_filters, stride=stride, kernel_size=kernel_size, padding=padding_d, bias=False))
            self.layers.append(nn.BatchNorm1d(num_filters))
            self.layers.append(nn.LeakyReLU())
        #self.layers.append(nn.Upsample(size=4233, mode='linear'))
        #self.layers.append(nn.Upsample(size=8192, mode='linear'))
        self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=False))
        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x 


class DnCNN_Res(nn.Module):
    def __init__(self, num_channels=1, num_layers=17, kernel_size=3, stride=1, num_filters=64, dilation_rate=3):
        super(DnCNN_Res, self).__init__()

        padding = int((kernel_size-1)/2)
        padding_d = math.floor((kernel_size + (kernel_size-1) * (dilation_rate-1))/2)

        # create module list
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_filters, kernel_size=kernel_size, padding=padding, bias=False))
        self.layers.append(nn.LeakyReLU())
        for i in range((num_layers-2)//2):
            self.layers.append(ResBlock( num_filters, kernel_size, padding_d, dilation_rate))
        self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)
                
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x 


class ResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, padding, dilation_rate):
        super(ResBlock, self).__init__()
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=padding, bias=False))
        self.layers.append(nn.BatchNorm1d(num_filters))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=padding, bias=False))
        self.layers.append(nn.BatchNorm1d(num_filters))
        self.layers.append(nn.LeakyReLU())
        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        identity = x
        for layer in self.model:
            x = layer(x)
        return nn.LeakyReLU()(identity+x)
