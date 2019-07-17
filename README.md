# SC-LSTM

### Shared Cell LSTM
This repository contains the code used for multi-task sequence labeling experiments in [SC-LSTM: Learning Task-Specific Representation in Multi-task Learning for Sequence Labeling](https://www.aclweb.org/anthology/N19-1249) paper, partly forked from the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
If you use this code or our results in your research, we'd appreciate if you cite our apper as following:


```
@inproceedings{lu2019sc,
  title={SC-LSTM: Learning Task-Specific Representations in Multi-Task Learning for Sequence Labeling},
  author={Lu, Peng and Bai, Ting and Langlais, Philippe},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={2396--2406},
  year={2019}
}
```
## Requirements
Python 3.5+, PyTorch 0.4 and Allennlp (For ELMo embeddings) are required for the current repo.

## Steps

1. Preprossing data: NER {CoNLL2003}, Chunking {CoNLL2000} and POS {UD English POS}
    The files should formatted as same as the CoNLL 2000 dataset.
    For example,
    ```

    Rockwell NNP B-NP
    International NNP I-NP
    Corp. NNP I-NP
    's POS B-NP
    Tulsa NNP I-NP
    unit NN I-NP
    said VBD B-VP
    it PRP B-NP
    signed VBD B-VP
    a DT B-NP
    tentative JJ I-NP
    agreement NN I-NP
    extending VBG B-VP
    its PRP$ B-NP
    contract NN I-NP
    with IN B-PP
    Boeing NNP B-NP
    Co. NNP I-NP
    ```

    Then put CoNLL2003/CoNLL2000/Unidep datasets in ```./data/``` , put embedding files like ```glove.6B.100d.txt``` in  ```./data/emb```
    You can change training groups by delete specific tasks in the dict:datasets_config of file ```config_uni.py```

2. Train the model:
          command: ```python main_uni.py```
   The default setting is in ```config_uni.py``` file.
          
