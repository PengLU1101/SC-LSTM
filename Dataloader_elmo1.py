import nltk
import torch
import torch.utils.data as data
import itertools

import pickle
USE_CUDA = torch.cuda.is_available()

def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

def multitask_dataloader(path, num_task, batch_size=8):
    data_dict = read_pkl(path)
    datasets = data_dict["datasets"]
    mappings = data_dict["mappings"]
    num_feat = len(data_dict["mappings"]["casing"])
    num_voc = len(data_dict["embeddings"])
    num_char = len(data_dict["mappings"]["characters"])
    embeddings = data_dict["embeddings"]
    tgt_dict = {}

    for key, value in data_dict["mappings"].items():
        if key.endswith("BIO") or key.endswith("POS"):
        #if key.endswith("IOBES") or key.endswith("POS"):
            tgt_dict[key.split("_")[0]] = value
    tgt_size = []
    data = data_dict["data"]
    assert num_task == len(datasets)
    task_holder = []
    listname = ['trainMatrix', 'devMatrix', 'testMatrix']
    task2id, id2task = {}, {}
    for data_name, lists in data.items():
        task2id[data_name] = len(task2id)
        id2task[len(id2task)] = data_name
        train_loader = get_loader(lists['trainMatrix'], mappings, data_name, batch_size)
        dev_loadaer = get_loader(lists['devMatrix'], mappings, data_name, batch_size*10)
        test_loadaer = get_loader(lists['testMatrix'], mappings, data_name, batch_size*10)
        task_holder.append({"train": train_loader,
                             "dev": dev_loadaer,
                             "test": test_loadaer})

    return task_holder, task2id, id2task, num_feat, num_voc, num_char, tgt_dict, embeddings


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_list, name, mappings):
        """Reads source and target sequences from pkl files.
        Args:
            data_dict:[dict] file contains all tasks' data, mappings, embeddings
        """
        self.name = name
        self.data_list = data_list

        self.mappings = mappings

        self.num_total_pairs = len(self.data_list)


    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        name2out = {"conll2003": "NER_BIO", "conll2000": "chunk_BIO", "unidep": "POS"}

        src_seq = self.data_list[index]["tokens"]

        trg_seq = self.data_list[index][name2out[self.name]]
        src_feats = self.data_list[index]["casing"]
        src_chars = self.data_list[index]["characters"]
        src_tokens = self.data_list[index]['raw_tokens']
        return src_seq, trg_seq, src_feats, src_chars, src_tokens

    def __len__(self):
        return self.num_total_pairs

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        src_mask_w: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        src_mask_s: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_seqs: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_mask_w: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_mask_s: torch tensor of shape (batch_size, padded_num_sent, padded_length).
    """
    def merge(sequences):
        num_sents = [len(sent) for sent in sequences]
        padded_seqs = torch.zeros(len(sequences), max(num_sents)).long()
        mask_s = torch.zeros(len(sequences), max(num_sents)).long()
        for i, sent in enumerate(sequences):
            end = num_sents[i]
            mask_s[i, :end] = 1
            padded_seqs[i, :end] = torch.LongTensor(sent[:end])
        mask_s = mask_s.float()
        if USE_CUDA:
            padded_seqs = padded_seqs.cuda()
            mask_s = mask_s.cuda()
        return padded_seqs, mask_s
    def merge_char(sequences):
        num_sents = [len(sent) for sent in sequences]
        num_chars = [[len(word) for word in sent] for sent in sequences]
        max_seq_length = max(num_sents)
        max_char_length = max(itertools.chain.from_iterable(num_chars))


        padded_seqs = torch.zeros(len(sequences), max_seq_length, max_char_length).long()
        for i, sent in enumerate(sequences):
            end = num_sents[i]
            for j, seq in enumerate(sent):
                endd = num_chars[i][j]
                padded_seqs[i, j, :endd] = torch.LongTensor(seq[:endd])
        if USE_CUDA:
            padded_seqs = padded_seqs.cuda()
        return padded_seqs

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, tgt_seqs, src_feats, src_chars, src_tokens = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    tgt_list = tgt_seqs
    src_seqs, src_masks = merge(src_seqs)
    tgt_seqs, tgt_masks = merge(tgt_seqs)
    src_feats, _ = merge(src_feats)
    src_chars = merge_char(src_chars)

    return src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars, tgt_list, src_tokens


def get_loader(data_list, mappings, name, batch_size=1):
    """Returns data loader for custom dataset.

    Args:
        pkl_path: pkl file path for source domain.
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(data_list, name, mappings)

    # data loader for custome dataset
    # this will return (src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    return data_loader

def test():
    pkl_path = "./pkl/ontonotes_conll2003_embeddings.pkl"
    holder = multitask_dataloader(pkl_path, 2)
    ll = []
    for item in holder:
        ll.append(item["train"])
    x = ll[0] + ll[1]
    print(len(x))
    print(len(ll[0]))
    print(len(ll[1]))
    """
    data_iter = iter(item["train"])
    src_seqs, src_mask_s, tgt_seqs, tgt_mask_s = next(data_iter)
    print(src_seqs.size())
    print(src_mask_s.size())
    print(tgt_seqs.size())
    print(tgt_mask_s.size())
    """
if __name__ == '__main__':
    test()
