## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
if __name__ == '__main__':
    t = TextParser()
    test =t.get_word_indices("fine",18)
    print(test)