import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HW(nn.Module):
    def __init__(self, dim, num_layers=1, dropout=0.5):
        super(HW, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.trans_list = nn.ModuleList()
        self.gate_list = nn.ModuleList()
        self.Dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            tmp_t = nn.Linear(dim, dim)
            tmp_g = nn.Linear(dim, dim)
            self.trans_list.append(tmp_t)
            self.gate_list.append(tmp_g)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.init_linear(self.trans_list[i])
            self.init_linear(self.gate_list[i])

    def init_linear(self, input_linear):
        #stdv = 1.0 / np.sqrt(self.dim)  
        stdv = np.sqrt(3.0 / self.dim)  

        input_linear.weight.data.uniform_(-stdv, stdv)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def forward(self, x):
        for i in range(self.num_layers):
            if i == 0:
                x = self.Dropout(x)
            g = F.sigmoid(self.gate_list[i](x))
            h = F.relu(self.trans_list[i](x))
            x = g * h + (1 - g) * x

        return x
