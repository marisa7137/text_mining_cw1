import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep

class Model(torch.nn.Module):
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
        #print(out)
        #print(out.shape) #shape:torch.Size([545, 18, 30])
        out = self.sen_rep(out)
        #print("sen_rep:",out.shape)  #([545, 30])
        #print("sen_rep:",out)
        out = self.fc1(out)
        #print("fc1 size",out.shape) #[545, 30])
        out = self.af1(out)
        #print("af1 size",out.shape) #([545, 30])
        out = self.fc2(out)
        #print("fc2 size",out.shape) #([545, 50])
        out = self.af2(out)
        #print("af2 size",out.shape) #([545, 50])
        #print(out)
        #print("torch mean:",torch.mean(out,dim=1).shape) #dim=1说明每一行求平均
        return out
