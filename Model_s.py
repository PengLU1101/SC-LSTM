import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from allennlp.modules.elmo import Elmo, batch_to_ids
import os
from Layer.conditional_random_field import *
from Layer.SCLSTM_Layer import *
from Layer.Embedding_Layer import *
from Layer.Highway_Layer import *
import pickle
USE_CUDA = torch.cuda.is_available()
USE_CASE = False
HIGH_WAY = False


class Char_CNN(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, padding, dropout):
        super(Char_CNN, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.Dropout = nn.Dropout(dropout)
        self.cnn = nn.Conv1d(in_channels=d_in, 
                             out_channels=d_out, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.relu = nn.ReLU()
    def forward(self, input):
        """
        Args:
            input[FloatTensor]: batch x seq_len x char_len x d_in
        Outs:
            outs[FloatTensor]: batch x seq_len x d_out
        """
        batch_size, seq_len, char_len, d_in = input.size()
        assert self.d_in == d_in
        input = input.view(batch_size*seq_len, char_len, -1).transpose(1, 2).contiguous()
        char_cnn_out = self.cnn(self.Dropout(input))
        char_cnn_out = self.relu(char_cnn_out)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, seq_len, -1)
        return char_cnn_out



class SFM_Classifier(nn.Module):
    def __init__(self, d_in, d_out, dropout=0):
        super(SFM_Classifier, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout_p = dropout
        #self.activation = activation
        #define layers
        self.Dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, d_out)
        self.sfm = nn.LogSoftmax(dim=-1)

    def forward(self, feats, tags, masks):
        """
        Args: 
            feats[FloatTensor]: (seq_len, batch_size, d_in)
        Outs:
            outs[FlaotTensor]: (seq_len, batch_size, d_out)
        """
        seq_len, batch_size, d_in = feats.size()
        assert d_in == self.d_in

        out = self.fc(self.Dropout(feats).contiguous().view(batch_size*seq_len, -1))
        #if self.activation:
        #    out = F.relu(out)
        out = self.sfm(out.contiguous().view(seq_len, batch_size, -1)).transpose(1, 0)
        return out.contiguous() 

    def inference(self, feats, mask):
        seq_len, batch_size, d_in = feats.size()
        assert d_in == self.d_in

        out = self.fc(self.Dropout(feats))
        #if self.activation:
        #    out = F.relu(out)
        out = self.sfm(out)
        scores, preds = torch.max(out.transpose(1,0), dim=-1)
        preds = preds * mask.long()
        return preds 

class CRF_Classifier(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(CRF_Classifier, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout_p = dropout
        #define layers
        self.Dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, d_out)
        self.crf = ConditionalRandomField(self.d_out)

    def forward(self, feats, tags, masks):
        """
        Args: 
            feats[FloatTensor]: (seq_len, batch_size, d_in)
        Outs:
            outs[FlaotTensor]: (batch_size, seq_len, d_out)
        """
        seq_len, batch_size, d_in = feats.size()
        assert d_in == self.d_in
        logits = self.Dropout(feats)
        logits = self.fc(feats.transpose(1, 0).contiguous()) # batch x seq_len x d_out
        
        loss = self.crf(logits, tags, masks)
        return loss
    def inference(self, feats, mask):
        logits = self.fc(feats.transpose(1, 0).contiguous()) # batch x seq_len x d_out
        decoded = self.crf.viterbi_tags(logits, mask)
        return decoded



class Neural_Tagger_cnn(nn.Module):
    def __init__(self, Word_embeddings, Feature_embeddings, Char_embeddings,
        ShareRNN, CNN, Classifierlist, concat_flag, elmo=None, elmo_flag=False):
        super(Neural_Tagger_cnn, self).__init__()
        self.Word_embeddings = Word_embeddings
        if USE_CASE:
            self.Feature_embeddings = Feature_embeddings
        self.Char_embeddings = Char_embeddings
        self.ShareRNN = ShareRNN
        self.cnn = CNN
        self.Classifierlist = Classifierlist
        self.concat_flag = concat_flag
        #self.regulizaion_flag = regulizaion_flag
        if HIGH_WAY:
            self.hw_char = HW(CNN.d_out, 3)

        if elmo_flag:
            self.elmo_flag = elmo_flag
            self.elmo = elmo
            if USE_CUDA:
                self.elmo = self.elmo.cuda()

    def forward(self, src_seqs, src_masks, src_feats, src_chars, 
               tgt_seqs, tgt_masks, 
               idx, src_tokens=None):

        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        feats, h_p, h_s = self.encode(src_words, src_masks, src_feats, src_chars, taskidx, src_tokens)
        loss = self.Classifierlist[taskidx](feats, tgt_seqs, src_masks)
        return loss, h_p, h_s

    def encode(self, src_words, src_masks, src_feats, src_chars, taskidx, src_tokens=None):

        batch_size, seq_len = src_words.size()
        word_embeds = self.Word_embeddings(src_words)
        if USE_CASE:
            feat_embeds = self.Feature_embeddings(src_feats)
        batch_char, seq_char_len, char_len = src_chars.size()
        assert batch_size == batch_char
        assert seq_len == seq_char_len
        src_chars = src_chars.view(batch_size*seq_len, -1)
        char_embeds = self.Char_embeddings(src_chars)
        char_embeds = char_embeds.view(batch_size, seq_len, char_len, -1)
        char_cnn_out = self.cnn(char_embeds)
        #print(word_embeds)
        if HIGH_WAY:
            char_cnn_out = self.hw_char(char_cnn_out)

        if self.elmo_flag:
            character_ids = batch_to_ids(src_tokens)
            character_ids = character_ids.cuda()
            #print(type(character_ids))
            elmo_embeddings = self.elmo(character_ids)
            inputs = torch.cat((word_embeds, char_cnn_out, elmo_embeddings['elmo_representations'][-1]), dim=-1)
        else:
            inputs = torch.cat((word_embeds, char_cnn_out), dim=-1)
        if USE_CASE:
            inputs = torch.cat((inputs, feat_embeds), dim=-1)

        h_f, c_f, final_list = self.ShareRNN(inputs, src_masks, taskidx)

        ####
        if self.concat_flag:
            concat_hid = torch.cat((final_list[taskidx][:, -1, :, :], h_f[:, -1, :, :]), dim=-1)
        else:
            concat_hid = final_list[taskidx][:, -1, :, :] + h_f[:, -1, :, :]
        #outs = self.Classifierlist[taskidx](concat_hid, tgt_seqs, src_masks)
        h_p = final_list[taskidx]
        return concat_hid, h_p, h_f
    #def decode(self, feats, tgt_seqs, src_masks, tgt_seqs, taskidx):
    #    outs = self.Classifierlist[taskidx](feats, tgt_seqs, src_masks)
        
    def predict(self, src_seqs, src_masks, src_feats, src_chars,
               tgt_seqs, tgt_masks, idx, src_tokens=None):
        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        feats, _, _ = self.encode(src_words, src_masks, src_feats, src_chars, taskidx, src_tokens)
        preds = self.Classifierlist[taskidx].inference(feats, src_masks)
        return preds

class Neural_Tagger(nn.Module):
    def __init__(self, Word_embeddings, Feature_embeddings,
        ShareRNN, Classifierlist, concat_flag):
        super(Neural_Tagger, self).__init__()
        self.Word_embeddings = Word_embeddings
        self.Feature_embeddings = Feature_embeddings
        self.ShareRNN = ShareRNN
        self.Classifierlist = Classifierlist
        self.concat_flag = concat_flag
        #self.regulizaion_flag = regulizaion_flag

    def forward(self, src_seqs, src_masks, src_feats,
               tgt_seqs, tgt_masks, 
               idx):

        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        feats, h_p, h_s = self.encode(src_words, src_masks, src_feats, taskidx)
        loss = self.Classifierlist[taskidx](feats, tgt_seqs, src_masks)
        return loss, h_p, h_s

    def encode(self, src_words, src_masks, src_feats, taskidx):

        batch_size, seq_len = src_words.size()
        word_embeds = self.Word_embeddings(src_words)
        feat_embeds = self.Feature_embeddings(src_feats)
        inputs = torch.cat((word_embeds, feat_embeds), dim=-1)

        h_f, c_f, final_list = self.ShareRNN(inputs, src_masks)

        ####
        if self.concat_flag:
            concat_hid = torch.cat((final_list[taskidx], h_f), dim=-1)
        else:
            concat_hid = final_list[taskidx] + h_f
        #outs = self.Classifierlist[taskidx](concat_hid, tgt_seqs, src_masks)
        h_p = final_list[taskidx].detach()
        h_ff = h_f.detach() 
        return concat_hid, h_p, h_ff
    #def decode(self, feats, tgt_seqs, src_masks, tgt_seqs, taskidx):
    #    outs = self.Classifierlist[taskidx](feats, tgt_seqs, src_masks)
        
    def predict(self, src_seqs, src_masks, src_feats,
               tgt_seqs, tgt_masks, idx):
        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        feats, _, _ = self.encode(src_words, src_masks, src_feats, taskidx)
        preds = self.Classifierlist[taskidx].inference(feats, src_masks)
        return preds
    

def build_model(para):
    emsize = para["d_emb"]
    d_hid = para["d_hid"]
    d_feat = para["d_feat"]
    n_layers = para["n_layers"]
    dropout = para["dropout"]
    n_feats = para["n_feats"]
    n_vocs = para["n_vocs"]
    n_tasks = para["n_tasks"]
    crf_flag = para["crf"]
    out_size = para["out_size"]
    concat_flag = para["concat_flag"]
    print(para)
    Word_embeddings = Embedding_Layer(n_vocs, emsize)
    Feature_embeddings = Embedding_Layer(n_feats, d_feat)
    rnn = StackedLSTMCell(emsize+d_feat, d_hid//2, n_layers, dropout, share=n_tasks)
    ShareRNN = BiSLSTM(rnn)
    if concat_flag:
        d_in_fc = d_hid * 2
    else:
        d_in_fc = d_hid
    
    if crf_flag:
        Classifierlist = nn.ModuleList([CRF_Classifier(d_in_fc, d_out) for d_out in out_size])
    else:
        Classifierlist = nn.ModuleList([SFM_Classifier(d_in_fc, d_out) for d_out in out_size])
    model = Neural_Tagger(Word_embeddings, Feature_embeddings,
        ShareRNN, Classifierlist, concat_flag)
    if USE_CUDA:
        model = model.cuda()

    return model
    

def build_model_cnn(para):
    emsize = para["d_emb"]
    d_hid = para["d_hid"]
    d_feat = para["d_feat"]
    n_layers = para["n_layers"]
    dropout = para["dropout"]
    n_feats = para["n_feats"]
    n_vocs = para["n_vocs"]
    n_tasks = para["n_tasks"]
    crf_flag = para["crf"]
    out_size = para["out_size"]
    n_chars = para["n_chars"] ###########
    d_char_emb = para["d_char_emb"] ###########
    d_char = para["d_char"] ###########
    kernel_size = para["kernel_size"] ######
    padding = para["padding"] ######

    concat_flag = para["concat_flag"]
    elmo_flag = para["use_elmo"]

    print(para)
    Word_embeddings = Embedding_Layer(n_vocs, emsize)
    Feature_embeddings = Embedding_Layer(n_feats, d_feat)
    Char_embeddings = Embedding_Layer(n_chars, d_char_emb)
    if elmo_flag:
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        elmo = Elmo(options_file, weight_file, 2, dropout=0) 
        rnn = StackedLSTMCell(emsize+d_char+1024, d_hid//2, n_layers, dropout, share=n_tasks)
    else:
        elmo = None
        if USE_CASE:
            rnn = StackedLSTMCell(emsize+d_feat+d_char, d_hid//2, n_layers, dropout, share=n_tasks)
        else:
            rnn = StackedLSTMCell(emsize+d_char, d_hid//2, n_layers, dropout, share=n_tasks)

    ShareRNN = BiSLSTM(rnn)
    CNN = Char_CNN(d_char_emb, d_char, kernel_size, padding, dropout=dropout)
    if concat_flag:
        d_in_fc = d_hid * 2
    else:
        d_in_fc = d_hid
    
    if crf_flag:
        Classifierlist = nn.ModuleList([CRF_Classifier(d_in_fc, d_out, dropout) for d_out in out_size])
    else:
        Classifierlist = nn.ModuleList([SFM_Classifier(d_in_fc, d_out, dropout) for d_out in out_size])
    model = Neural_Tagger_cnn(Word_embeddings, Feature_embeddings, Char_embeddings,
        ShareRNN, CNN,
        Classifierlist, concat_flag, elmo, elmo_flag)
    if USE_CUDA:
        model = model.cuda()

    return model

def save_model(path, model, para):
    model_path = os.path.join(path, 'model.pt')
    torch.save(model.state_dict(), model_path)
    para_path = os.path.join(path, 'para.pkl')
    with open(para_path, "wb") as f:
        pickle.dump(para, f)

def read_model(path, model):
    model_path = os.path.join(path, 'model.pt')
    model.load_state_dict(torch.load(model_path))
    return model


def test():
    pass



if __name__ == '__main__':
    test()
