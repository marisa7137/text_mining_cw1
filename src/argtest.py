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
    config.read('src/bilstm.config')
    t_train = TextParser(pathfile=config.get("param", "train_5500"), tofile=True, stopwords_pth=config.get("param", "stop_words"), fine_label_pth=config.get(
        "param", "fine_label"), coarse_label_pth=config.get("param", "coarse_label"), vocab_pth=config.get("param", "vocab"))
    test = t_train.get_word_indices("fine", dim=20, from_file=True)
    print(t_train.fine_pair[15])

   
    
    
    #clearprint(test)
        

