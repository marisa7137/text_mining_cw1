## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
import bilstm_train
import bilstm_test
import BagOfWords_test
import BagofWords_train
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
#from bilstm import Model
from BagofWords import Model
import torch.optim as optim
import sys
import os
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/BagofWords.config")

if __name__ == '__main__':
    # coarse
    # fine
    class_type = "fine"

    t_train = TextParser(pathfile=config.get("param","path_train"))
    #t_train=TextParser("data/train.txt")
    train_data = t_train.get_word_indices(class_type, dim=18)
    t_test = TextParser(pathfile=config.get("param","path_dev"))
    #t_test=TextParser("data/test.txt")
    test_data = t_test.get_word_indices(class_type, dim=18)
    #print(train_data)
    #print(test_data)


    if class_type == "coarse":
        # # train the model
        BagofWords_train.train(t_train, train_data, num_classes=6)

        # test the model
        BagOfWords_test.test(test_data, num_classes=6)

    else:
        # # train the model
        BagofWords_train.train(t_train, train_data, num_classes=50)

        # test the model
        BagOfWords_test.test(test_data, num_classes=50)
