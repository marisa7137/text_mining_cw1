import torch
import torch.nn as nn
from word_embedding import Word_Embedding

class Sentence_Rep(nn.Module):
    def __init__(self, bow: bool, embedding_dim, hidden_dim_bilstm):
        super().__init__()
        self.hidden_dim_bilstm = hidden_dim_bilstm
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim_bilstm, bidirectional=True, batch_first=True)
        # self.fc = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=embedding_dim)
        self.bow = bow


    def forward(self, word_vecs):
        if self.bow:
            out = word_vecs
            return out
        else:
            bilstm_out, _ = self.bilstm(word_vecs)
            back = bilstm_out[:, 0, self.hidden_dim_bilstm:]
            forward = bilstm_out[:, -1, :self.hidden_dim_bilstm]
            out = torch.cat((forward, back), dim=1)
            # print(out.shape)
            return out

