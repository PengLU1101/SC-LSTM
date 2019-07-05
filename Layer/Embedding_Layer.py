import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class Embedding_Layer(nn.Module):
    def __init__(self, vocab, d_emb, scope=.25):
        super(Embedding_Layer, self).__init__()
        self.lut = nn.Embedding(vocab, d_emb)
        self.d_emb = d_emb
        #scope = np.sqrt(1.0 / d_emb)
        scope = 1
        self.lut.weight.data.uniform_(-scope, scope)
        self.lut.weight.data[0] = torch.zeros(1, d_emb)

    def forward(self, x, norm_flag=True):
        """
        Arguments:
            x: [batch_size, seq_len] LongTensor
        Output:
            embeds: [batch, seq_len, d_emb] FloatTensor
        """
        embeds = self.lut(x)  * math.sqrt(self.d_emb) if norm_flag else self.lut(x)

        return embeds
    def apply_weights(self, weights, fine_tune_flag=True):
        if isinstance(weights, np.ndarray):
            self.lut.weight.data.copy_ (torch.from_numpy(weights))
        else:
            pass
            #self.lut.weight
        if not fine_tune_flag:
            for p in self.lut.parameters():
                p.requires_grad = False