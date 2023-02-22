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

def randomly_initialised_vectors(tokens=None,threshold=None):
    wordCountDict = dict(zip(*np.unique(tokens, return_counts=True)))
    for k in list(wordCountDict.keys()):  # 对字典a中的keys，相当于形成列表list
        if wordCountDict[k] < threshold:
            del wordCountDict[k]

    wordToIx = {}
    wordToIx['UNK'] = 0
    i = 1
    for key in wordCountDict.keys():
        wordToIx[key] = i
        i = i+1
    word_vectors = []
    for _ in wordToIx:
        word_vectors.append(np.random.random(18))
    word_vectors = np.array(word_vectors)
    return word_vectors,wordToIx

if __name__ == '__main__':
    t = TextParser(pathfile='../data/train_5500.label.txt')
    t.get_word_indices("coarse",18)
    print(t.indexed_sentence_pair[0])
    
    t = TextParser(pathfile=config.get("param","path_train"))
    print(randomly_initialised_vectors(t.words,3)[1])
    
    
    
    #clearprint(test)
        

