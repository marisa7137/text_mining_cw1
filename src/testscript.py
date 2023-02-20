## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
import model_train
import model_test
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
import sys
import os


if __name__ == '__main__':
    # coarse
    # fine
    class_type = "fine"

    t = TextParser()
    wi = t.get_word_indices(class_type, dim=18)

    # data size = len(wi) = train_indices + test_indices = 5452
    # train_size = 0.9*5452 = 4907
    train_indices = 4905
    # test_size = 0.1*5452 = 545
    test_indices = 545

    train_data = [wi[i] for i in range(train_indices)]
    test_data = [wi[i] for i in range(train_indices, train_indices + test_indices)]

    if class_type == "coarse":
        # # train the model
        model_train.train(t, train_data, num_classes=6)

        # test the model
        model_test.test(test_data, num_classes=6)

    else:
        # # train the model
        model_train.train(t, train_data, num_classes=50)

        # test the model
        model_test.test(test_data, num_classes=50)
