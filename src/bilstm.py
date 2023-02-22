import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep


class Model(nn.Module):
    '''
    This class that builds the BiLSTM model with a classifier.
    :param list pre_train_weight: the pre-trained weights
    :param int vocab_size: the size of vocabulary in text parser
    :param int embedding_dim: the embedding dimension (suggested 300 at least)
    :param bool from_pre_train: True if use the pre-trained wrights
    :param bool freeze: True if freeze the weights
    :param bool bow: False if builds BiLSTM
    :param int hidden_dim_bilstm: the hidden dimension of BiLSTM
    :param int hidden_layer_size: the hidden layer size of BiLSTM
    :param int num_of_classes: the number of classes, 6 or 50
    :return: a BiLSTM model with a classifier
    '''
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool, bow: bool, hidden_dim_bilstm, hidden_layer_size, num_of_classes):
        super().__init__()
        self.word_embedding = Word_Embedding(pre_train_weight=pre_train_weight, vocab_size=vocab_size, embedding_dim=embedding_dim, from_pre_train=from_pre_train, freeze=freeze)
        self.sen_rep = Sentence_Rep(bow=bow, embedding_dim=embedding_dim, hidden_dim_bilstm=hidden_dim_bilstm)
        self.fc1 = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=hidden_layer_size)
        self.af1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=num_of_classes)
        self.af2 = nn.LogSoftmax(dim=0)


    def forward(self, indexed_sentence):
        out = self.word_embedding(indexed_sentence)
        # print(out.shape)
        out = self.sen_rep(out)
        out = self.fc1(out)
        out = self.af1(out)
        out = self.fc2(out)
        out = self.af2(out)
        return out