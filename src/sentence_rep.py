import torch
import torch.nn as nn
from word_embedding import Word_Embedding

class Sentence_Rep(nn.Module):
    def __init__(self, bow: bool, embedding_dim, hidden_dim_bilstm):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim_bilstm, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=embedding_dim)
        self.bow = bow


    def forward(self, word_vecs):
        if self.bow:
            out = word_vecs
            return out
        else:
            bilstm_out, _ = self.bilstm(word_vecs)
            out = self.fc(bilstm_out)
            return out

