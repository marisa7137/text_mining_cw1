import torch
import torch.nn as nn
from word_embedding import Word_Embedding

class Sentence_Rep(nn.Module):
    def __init__(self, word_embedding: Word_Embedding, bow: bool, embedding_dim, hidden_dim_bilstm):
        super().__init__()
        self.embedding = word_embedding
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim_bilstm, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=embedding_dim)
        self.bow = bow


    def forward(self, sentence):
        sen_embedded = self.embedding(sentence)
        if self.bow:
            out = torch.mean(sen_embedded, dim=0)
            return out
        else:
            bilstm_out, _ = self.bilstm(sen_embedded)
            out = self.fc(bilstm_out)
            out = torch.mean(out, dim=0)
            return out

