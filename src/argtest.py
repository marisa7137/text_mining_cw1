## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
from word_embedding import Word_Embedding
from configparser import ConfigParser

# import the configure files


if __name__ == '__main__':
    config = ConfigParser()
    config.read('bilstm.config')
    t_test = TextParser(pathfile='../data/train_5500.label.txt', tofile=True)
    t_test.create_vocab_and_weight_from_pretrained_glove()
    # l = t_test.get_word_indices_from_glove(type='fine', dim=20)
    # print(l)
    
    
    
    #clearprint(test)
        

