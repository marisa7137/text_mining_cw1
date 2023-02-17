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


if __name__ == '__main__':
    t = TextParser()
    wi = t.get_word_indices(dim=100)
    # random_perm = np.random.permutation(len(wi))
    # test_size = int(0.1 * len(wi))                       ### change to a fixed number!!!!!!!!
    # train_indices = random_perm[test_size:]
    # test_indices = random_perm[:test_size]


    # data size = len(wi) = train_indices + test_indices = 5452
    # train_size = 0.9*5452 = 4907
    train_indices = 4907
    # test_size = 0.1*5452 = 545
    test_indices = 545

    train_data = [wi[i] for i in range(train_indices)]
    test_data = [wi[i] for i in range(train_indices, train_indices + test_indices)]

    # train the model
    # model_train.train(t, train_data)

    # test the model

    model_test.test(test_data, num_classes=len(t.labels))