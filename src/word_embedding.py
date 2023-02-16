import torch.nn as nn


class Word_Embedding(nn.Module):
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool):
        super().__init__()
        if from_pre_train:
            self.embedding = nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, word_indices):
        out = self.embedding(word_indices)
        return out
