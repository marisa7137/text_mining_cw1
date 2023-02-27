import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep

class Model(torch.nn.Module):
    '''
    This class that builds the BOW model with a classifier.
    :param list pre_train_weight: the pre-trained weights
    :param int vocab_size: the size of vocabulary in text parser
    :param int embedding_dim: the embedding dimension (suggested 300 at least)
    :param bool from_pre_train: True if use the pre-trained wrights
    :param bool freeze: True if freeze the weights
    :param bool bow: False if builds BOW
    :param int hidden_dim_bilstm: the hidden dimension of BOW
    :param int hidden_layer_size: the hidden layer size of BOW
    :param int num_of_classes: the number of classes, 6 or 50
    :return: a BOW model with a classifier
    '''
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool, bow: bool, hidden_dim_bilstm, hidden_layer_size, num_of_classes):
        super().__init__()
        
        self.word_embedding = Word_Embedding(pre_train_weight=pre_train_weight, vocab_size=vocab_size, embedding_dim=embedding_dim, from_pre_train=from_pre_train, freeze=freeze)
        self.sen_rep = Sentence_Rep(bow=bow, embedding_dim=embedding_dim, hidden_dim_bilstm=hidden_dim_bilstm)
        
        if from_pre_train:
            self.fc3=nn.Linear(in_features=300, out_features=hidden_layer_size)
        else:
            self.fc3=nn.Linear(in_features=embedding_dim, out_features=hidden_layer_size)
        self.af3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=hidden_layer_size, out_features=num_of_classes)
        self.af4 = nn.LogSoftmax(dim=0)
        self.norm = nn.BatchNorm1d(num_features=300) # BatchNorm2d only accepts 4D inputs while BatchNorm1d accepts 2D or 3D inputs
        self.dropout = nn.Dropout(p=0.1) # dropout
       


    def forward(self, indexed_sentence):
        # ------------ word embedding -------------
        out = self.word_embedding(indexed_sentence)
        # ------------ BOW embedding -------------
        out = self.sen_rep(out)
        # ------------ Classifier -------------
        # out = self.dropout(out)
        out = self.norm(out)
        out = self.fc3(out)
        out = self.af3(out)
        out = self.fc4(out)
        out = self.af4(out)
        return out
