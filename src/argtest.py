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
config = ConfigParser()
config.read("src/bow.config")

if __name__ == '__main__':
    t = TextParser(pathfile=config.get("param","path_train"))
    t.get_word_indices("coarse",18)
    print(t.indexed_sentence_pair[0])
