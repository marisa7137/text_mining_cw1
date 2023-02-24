import torch
import torch.nn as nn
from word_embedding import Word_Embedding


class Sentence_Rep(nn.Module):
    def __init__(self, bow: bool, embedding_dim, hidden_dim_bilstm):
        super().__init__()
        self.hidden_dim_bilstm = hidden_dim_bilstm
        self.bow = bow

    def forward(self, word_vecs):
        if self.bow:
            bow_vector = []
            for sentence in word_vecs:
                dim0, dim1 = sentence.shape
                word_sum = torch.zeros(dim1)
                for word in sentence:
                    word_sum = torch.add(word_sum, word)
                bow_sentence_tensor = torch.div(word_sum, dim0)
                bow_sentence_tensor = bow_sentence_tensor.detach()
                bow_vector.append(bow_sentence_tensor)
            out = torch.stack(bow_vector, 0)
            return out
        else:
            return 
