## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
import bilstm_train
import bilstm_test
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
import sys
import os
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/bilstm.config")

if __name__ == '__main__':
    # coarse
    # fine
    class_type = "fine"

    t_train = TextParser(pathfile=config.get("param","path_train"))
    train_data = t_train.get_word_indices(class_type, dim=18)
    t_test = TextParser(pathfile=config.get("param","path_dev"))
    test_data = t_test.get_word_indices(class_type, dim=18)
    


    if class_type == "coarse":
        # # train the model
        Model.train(t_train, train_data, num_classes=6)

        # test the model
        bilstm_test.test(test_data, num_classes=6)

    else:
        # # train the model
        bilstm_train.train(t_train, train_data, num_classes=50)

        # test the model
       # bilstm_test.test(test_data, num_classes=50)
