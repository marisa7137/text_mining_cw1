## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
from word_embedding import Word_Embedding
from configparser import ConfigParser
from configparser import ConfigParser
# import the configure files

if __name__ == '__main__':
    t = TextParser(pathfile='../data/train_5500.label.txt')
    t.get_word_indices("coarse",18)
    print(t.indexed_sentence_pair[0])
