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
    t_test = TextParser(pathfile=config.get("param","path_test"),tofile=False)
    test_data = t_test.get_word_indices("fine", dim=20, from_file=True)
    print(t_test.fine_pair)
    
    
    
    #clearprint(test)
        

