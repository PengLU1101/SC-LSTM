"""
updata at 2018-10-12
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import math
#from Embedding_Layer import *


USE_CUDA = torch.cuda.is_available()



class Share_lstm_cell(nn.Module):
    def __init__(self, d_in, d_hid, bias=True, layernorm=False, share=3):
        super(Share_lstm_cell, self).__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.bias = bias
        self.share = share
        self.num_gate = 4 + share
        self.weight_ih = nn.Parameter(torch.Tensor(self.num_gate * d_hid, d_in))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * d_hid, d_hid))
        self.weight_sh_list = nn.ParameterList([nn.Parameter(torch.Tensor(d_hid, d_hid)) for i in range(share)])
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(self.num_gate * d_hid))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * d_hid))
            self.bias_sh_list = nn.ParameterList([nn.Parameter(torch.Tensor(d_hid)) for i in range(share)])
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_sh', None)
        self.reset_parameters()
        self.share_num = float(share)
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.d_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, input, h_minors, c_minors, s_minors, mask, index):
        """

        args:
            input[FloatTensor]: batch x d_emb
            h_minors, c_minors[FloatTensor]: batch x d_hid
            mask[FloatTensor]: batch x 1
        return:
            h, c_t[FloatTensor]: batch x d_hid
            s[list of FloatTensor]: [batch x d_hid] * self.share

        """
        mask = mask.unsqueeze(1)
        assert mask.dim() == 2
        tmp = torch.addmm(self.bias_ih,
                          input,
                          self.weight_ih.transpose(1, 0))

        tmp2 = torch.addmm(self.bias_hh,
                           h_minors,
                           self.weight_hh.transpose(1, 0))
        tmp3_list = [torch.addmm(self.bias_sh_list[idx],
                                 s_minors[idx],
                                 self.weight_sh_list[idx].transpose(1, 0)) for idx in range(len(s_minors))]
        tmp3 = torch.cat(tmp3_list, dim=1)
        tmp_in, tmp_s = torch.split(tmp, 4 * self.d_hid, dim=1)
        ifo, c_hat = torch.split(tmp_in+tmp2, 3 * self.d_hid, dim=1)
        i, f, o = torch.split(torch.sigmoid(ifo), self.d_hid, dim=1)
        c_t = (f * c_minors + i * torch.tanh(c_hat)) * mask
        h = o * torch.tanh(c_t) * mask

        b = torch.split(torch.sigmoid(tmp_s+tmp3), self.d_hid, dim=1)
        #s = [bb * torch.tanh(c_t) * mask for bb in b]
        #s = [(bb + (1 - o)) / 2 * torch.tanh(c_t) * mask for bb in b]
        #s = [(bb * (1 - o)) * torch.tanh(c_t) * mask for bb in b]
        sum_s = 0
        s_list = [0] * self.share
        for i, x in enumerate(b):
            for m in s_list:
                sum_s += m
            s = (x + 1 + i - o - sum_s) / (1 + i) * torch.tanh(c_t) * mask
            s_list[i] = s
        return h, c_t, s_list

class StackedLSTMCell(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, dropout, share):
        super(StackedLSTMCell, self).__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.Dropout = nn.Dropout(dropout)
        self.share = share
        self.layers = nn.ModuleList([Share_lstm_cell(d_in, d_hid, share=share)])
        if n_layers > 1:
            for i in range(n_layers-1):
                self.layers.append(Share_lstm_cell(d_hid, d_hid, share=share))

    def forward(self, input, h_0, c_0, s_0, mask, idx):
        h_1, c_1, s_1 = [], [], []
        inputs = self.Dropout(input)
        
        #for i, layer in enumerate(self.layers):
        for i in range(self.n_layers):
            s_0_i = [x[i] for x in s_0]
            h_1_i, c_1_i, s_1_i = self.layers[i](inputs, h_0[i], c_0[i], s_0_i, mask, idx)
            
            inputs = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]
            if i == 1:
                s_1_i = [x[0] + x[1] for x in zip(s_1, s_1_i)]

            s_1 += s_1_i
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        s_list = [torch.stack(s_1[i::self.share]) for i in range(self.share)]

        return input, h_1, c_1, s_list
    def init_hid(self, batch_size):
        h, c = (Variable(torch.zeros(self.n_layers, batch_size, self.d_hid)) for i in range(2))
        return (h.cuda(), c.cuda()) if USE_CUDA else (h, c)
    def init_s(self, batch_size):
        s = [Variable(torch.zeros(self.n_layers, batch_size, self.d_hid)) for i in range(self.share)]
        s_list = [x.cuda() if USE_CUDA else x for x in s]
        return s_list

class BiSLSTM(nn.Module):
    """docstring for BiKBLSTM"""
    def __init__(self, RNN):
        super(BiSLSTM, self).__init__()
        self.RNN = RNN
        self.RNN_r = copy.deepcopy(RNN)
        self.n_direction = 2 

    def forward(self, inputs, mask, idx):
        """
        args:
            inputs[FloatTensor]: batch x seq_len x d_em
            mask[FlaotTensor]: batch x seq_len
        """
        batch_size, seq_len, emsize = inputs.size()
        inputs = inputs.transpose(1, 0)
        mask = mask.transpose(1, 0)
        h_list, c_list, s_lists = [], [], []
        h_r_list, c_r_list, s_r_lists = [], [], []

        h, c = self.RNN.init_hid(batch_size)
        h_r, c_r = self.RNN_r.init_hid(batch_size)

        s_list = self.RNN.init_s(batch_size)
        s_r_list = self.RNN_r.init_s(batch_size)

        for i in range(seq_len):
            _, h, c, s_list = self.RNN(inputs[i], h, c, s_list, mask[i], idx)
            """
            h_list += [h[-1]]
            c_list += [c[-1]]
            s_lists += [s[-1] for s in s_list]
            """
            h_list += [h]
            c_list += [c]
            s_lists += [s for s in s_list]
            _, h_r, c_r, s_r_list = self.RNN_r(inputs[seq_len-i-1], h_r, c_r, s_r_list, mask[seq_len-i-1], idx)
            h_r_list += [h_r]
            c_r_list += [c_r]
            s_r_lists += [s_r for s_r in s_r_list[::-1]]

        h_f = torch.cat((torch.stack(h_list, dim=0), torch.stack(h_r_list[::-1], dim=0)), dim=-1)
        c_f = torch.cat((torch.stack(c_list, dim=0), torch.stack(c_r_list[::-1], dim=0)), dim=-1)

        s_f_list = [torch.cat((itm[0], itm[1]), dim=-1) for itm in zip(s_lists, s_r_lists[::-1])]
        final_list = [torch.stack(s_f_list[i::self.RNN.share]) for i in range(self.RNN.share)]
        return h_f, c_f, final_list

def test():
    inputs = Variable(torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [3, 4 ,5, 0, 0]]))
    mask = torch.gt(inputs, 0).float()
    emblayer = Embedding_Layer(10, 100)
    rnn = StackedLSTMCell(100, 20, 2, 0.5, share=2)
    model = BiSLSTM(rnn)
    if USE_CUDA:
        inputs = inputs.cuda()
        mask = mask.cuda()
        model = model.cuda()
        emblayer = emblayer.cuda()
    a, b, c = model(emblayer(inputs), mask)
    print(a.size())
    print(b.size())
    for s in c:
        print(s.size())

if __name__ == '__main__':
    test()
