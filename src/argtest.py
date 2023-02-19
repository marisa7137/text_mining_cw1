## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
from word_embedding import Word_Embedding
from configparser import ConfigParser
if __name__ == '__main__':
    
    config = ConfigParser()
    config.read("src/bow.config")
    config.sections()
    print(config.get("param","stop_words"))
    # length = []
    # for i in range(0,len(t.fine_pair)):
    #     length.append(len(t.fine_pair[i][1]))
    # print(max(length))
