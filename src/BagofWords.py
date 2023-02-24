import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep

class Model(torch.nn.Module):
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool, bow: bool, hidden_dim_bilstm, hidden_layer_size, num_of_classes):
        super().__init__()
        
        self.word_embedding = Word_Embedding(pre_train_weight=pre_train_weight, vocab_size=vocab_size, embedding_dim=embedding_dim, from_pre_train=from_pre_train, freeze=freeze)
        self.sen_rep = Sentence_Rep(bow=bow, embedding_dim=embedding_dim, hidden_dim_bilstm=hidden_dim_bilstm)

        # self.fc1 = nn.Linear(in_features=embedding_dim, out_features=hidden_layer_size)
        # self.af1 = nn.LeakyReLU(0.1)
        # self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=num_of_classes)
        # self.af2 = nn.LogSoftmax(dim=0)

        self.fc3 = nn.Linear(in_features=embedding_dim, out_features=num_of_classes)
        self.af3 = nn.LogSoftmax(dim=0)


        #self.norm = nn.BatchNorm1d(2*hidden_dim_bilstm) # BatchNorm2d only accepts 4D inputs while BatchNorm1d accepts 2D or 3D inputs
        self.dropout = nn.Dropout(p=0.2) # dropout
       


    def forward(self, indexed_sentence):
        out = self.word_embedding(indexed_sentence)
        print("word_embedding out shape:",out.shape)
        out = self.sen_rep(out)
        print("sen_rep out shape:",out.shape)
        # out = self.fc1(out)
        # out = self.af1(out)
        # out = self.fc2(out)
        # out = self.af2(out)

        out = self.dropout(out)
        out = self.fc3(out)
        out = self.af3(out)

        return out
