import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep
# hh

class Model(nn.Module):
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool, bow: bool, hidden_dim_bilstm, hidden_layer_size, num_of_classes):
        super().__init__()
        self.word_embedding = Word_Embedding(pre_train_weight=pre_train_weight, vocab_size=vocab_size, embedding_dim=embedding_dim, from_pre_train=from_pre_train, freeze=freeze)
        self.sen_rep = Sentence_Rep(bow=bow, embedding_dim=embedding_dim, hidden_dim_bilstm=hidden_dim_bilstm)
        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=hidden_layer_size)
        self.af1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=num_of_classes)
        self.af2 = nn.LogSoftmax(dim=0)


    def forward(self, indexed_sentence):
        out = self.word_embedding(indexed_sentence)
        out = self.sen_rep(out)
        out = self.fc1(out)
        out = self.af1(out)
        out = self.fc2(out)
        out = self.af2(out)
        return torch.mean(out, dim=1)