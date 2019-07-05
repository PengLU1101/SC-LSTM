import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from WeightNorm import WeightNormConv2d
import math

SCALE_WEIGHT = 0.5 ** 0.5

def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)
def reshape_transform(x):
    return torch.squeeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, dilation=1, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(input_size, 
                                     2 * input_size,
                                     kernel_size=(width, 1), 
                                     stride=(1, 1),
                                     padding=(width // 2 * (1 - nopad)*dilation, 0), 
                                     dilation=dilation)
        init.xavier_uniform(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * F.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2, all_layers_hid=False):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.all_layers_hid = all_layers_hid
        for i in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout, 2**i))

    def forward(self, x, hidden=None):
        if self.all_layers_hid:
            hid = []
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
            if self.all_layers_hid:
                hid.append(x)
        return x if not self.all_layers_hid else hid
def test():
    inputs = shape_transform(Variable(torch.randn(10, 50, 10)))
    model = StackedCNN(3, 10)
    out = model(inputs)
    print(out.size())

if __name__ == '__main__':
    test()