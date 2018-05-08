import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

Torch4 = True if float(torch.__version__.split(".")[1]) > 3 


class BiLSTM_Layer(nn.Module):
	def __init__(self, d_input, d_hid, n_layers, dropout):
		super(BiLSTM_Layer, self).__init__()
		self.d_input = d_input
		self.n_layers = n_layers
		self.dropout_p = dropout

		assert d_hid % 2 == 0
		self.n_direction = 2
		self.d_hid = d_hid // 2
		self.rnn = nn.LSTM(d_input, self.d_hid, n_layers, dropout, bidirectional=True)
	def forward(self, in_seqs, in_lens):
		"""
    	Arguments:
            in_seqs: [batch_size, seq_len, d_input] FloatTensor
            in_lens: [batch_size, seq_len] list
        Output:
        	outs: [batch, seq_len, d_hid] FloatTensor
		"""
		batch_size, seq_len, d_input = in_seqs.size()
		assert d_input == self.d_input

		packed_inputs = pack(in_seqs.transpose(1, 0), in_lens)
		h0, c0 = self.init_hid(batch_size)
		outs, *_ = self.rnn(packed_inputs, (h0, c0))
		outs, _ = unpack(outs, in_lens)
		assert outs.size(0) == batch_size
		return outs

