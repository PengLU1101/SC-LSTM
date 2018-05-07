import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
Torch4 = True if float(torch.__version__.split(".")[1]) > 3 

class Embedding_Layer(nn.Module):
    def __init__(self, d_emb, vocab):

        super(Embedding_Layer, self).__init__()
        self.lut = nn.Embedding(vocab, d_emb)
        self.d_emb = d_emb

    def forward(self, x, norm_flag=False):
    	"""
    	Arguments:
            x: [batch_size, seq_len] LongTensor
        Output:
        	embeds: [batch, seq_len, d_emb] FloatTensor
		"""
		embeds = self.lut(x) * math.sqrt(self.d_emb) if norm_flag else self.lut(x)

        return embeds
    def apply_weights(self, weights, fine_tune_flag=False):
    	if isinstance(weights, np.array):
    		self.lut.weight.data.copy_ (torch.from_numpy(weights))
    	else:
    		pass
    		#self.lut.weight
    	if not fine_tune_flag:
    		for p in self.lut.parameters():
    			p.requires_grad = False

def BiLSTM_Layer(nn.Module):
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

class CRF_Layer(nn.Module):
    def __init__(self, tagset_size):
        super(CRF_Layer, self).__init__()
        self.tagset_size = tagset_size
        self.n_labels = n_labels = tagset_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))
        self.transitions.data.uniform_(0, 1)

    def reset_parameters(self):
        I.normal(self.transitions.data, 0, 1)

    def forward(self,
                logits,
                lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = torch.Tensor(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = Variable(alpha)
        if USE_CUDA:
            alpha = alpha.cuda()
        c_lens = lens.clone()
        logits_t = logits.transpose(1, 0).contiguous()
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1
        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self,
                       logits,
                       lens):
        batch_size, seq_len, n_labels = logits.size()
        vit = torch.Tensor(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = Variable(vit)
        if USE_CUDA:
            vit = vit.cuda()
        c_lens = lens.clone()
        pointers = []
        logits_t = logits.transpose(1, 0).contiguous()
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)
            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit
            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.stop_idx ].unsqueeze(0).expand_as(vit_nxt)
            c_lens = c_lens - 1
        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))
        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self,
                         labels,
                         lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()
        labels_ext = Variable(torch.Tensor(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        if USE_CUDA:
            labels_ext = labels_ext.cuda()
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext.long()
        labels = labels_ext
        trn = self.transitions
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)

        trn_scr = trn_scr.squeeze(-1)
        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

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