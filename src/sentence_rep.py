import torch
import torch.nn as nn


class Sentence_Rep(nn.Module):
    """
    Class: Sentence_Rep Class to initial the word representation of bag of words
    """

    def __init__(self, bow: bool, embedding_dim, hidden_dim_bilstm):
        super().__init__()
        self.hidden_dim_bilstm = hidden_dim_bilstm
        self.bow = bow

    def forward(self, word_vecs):
        """
        Function: 
            addes up each word vector in the sentence list and take the mean (average of the words in a sentence)
        Args:
            self(text parser),
            word_vecs: word vectors
        Return:
            return a out with [batch dim, word embediing]
        """
        if self.bow:
            bow = []
            for sentence in word_vecs:
                # sen_len, embedding_dim = sentence.shape
                sen_vec = []
                for word in sentence:
                    if torch.count_nonzero(word) > 0:
                        sen_vec.append(word)
                sen_vec = torch.stack(sen_vec, dim=0)
                sen_vec = torch.mean(sen_vec, dim=0)
                bow.append(sen_vec)
            out = torch.stack(bow, 0)
            return out
        else:
            return
