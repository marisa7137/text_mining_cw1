import torch
class BagOfWords(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.EmbeddingBag(vocab_size, 10)
        self.fc = torch.nn.Linear(10, 4)

    def forward(self, text):
        x = self.embedding(text, torch.zeros(text.shape, dtype=torch.long))
        x = self.fc(x)
        return x
