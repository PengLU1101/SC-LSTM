import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from CNN_Layer import *
from CRF_Layer import *
from RNN_Layer import *

Torch4 = True if float(torch.__version__.split(".")[1]) > 3 


class Neural_Tagger(nn.Module):
    def __init__(self, Word_embeddings, Feature_embeddings,
        ShareRNN, RNNlist, FNNlist, Classifierlist):
        self.Word_embeddings = Word_embeddings
        self.Feature_embeddings = Feature_embeddings
        self.ShareRNN = ShareRNN
        self.RNNlist = RNNlist
        self.FClist = FNNlist
        self.Classifierlist = Classifierlist

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        taskidx = data["task"]
        src_words = data["src_words"]
        src_feats = data["src_feats"]
        src_lens = data["src_length"]
        src_mask = data["src_mask"]
        #src_char = data["src_chars"] TO DO
        tgts = data["tgts"]

        batch_size, seq_len = src_words.size()
        word_embeds = self.Word_embeddings(src_words)
        feat_embeds = self.Feature_embeddings(src_feats)
        inputs = torch.cat((word_embeds, feat_embeds), dim=2)

        outs_share = self.ShareRNN(inputs, src_lens)
        outs_spc = self.RNNlist[taskidx](inputs, src_lens)
        out = torch.cat((outs_spc, outs_share[0], outs_share[1]), dim=2)
        logits = self.FClist[taskidx](out.contiguous().view(batch_size*seq_len, -1))
        return logits.contiguous().view(batch_size, seq_len, -1), tgts, src_mask, taskidx
    
    def decode(self, logits, tgts, src_mask, taskidx):
        batch_size, seq_len, d_hid = logits.size()
        return self.Classifierlist[taskidx](logits, tgts, src_mask)

    def predict(self, logits, tgts, src_mask, taskidx):
        return self.Classifierlist[taskidx].inference(logits, tgts, src_mask)


def build_model(para_dict):
    pass

para_dict = {
    "Word_embeddings":{"vocab_size":
                       "d_emb":
                       "padding_idx":
                       }
    "Feature_embeddings":{"feat_size":
                          "d_emb":
                          "padding_idx":
                          }
    "ShareRNN":{"d_in":
                "d_hid":
                ""}
}