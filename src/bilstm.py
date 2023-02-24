'''
Some notes:
1. For activation functions:
        The acc: tanh > LeakyReLU(0.1) > GELU()
        The speed of converging: GELU > tanh > LeakyReLU(0.1)
        -> All Overfitting
        -> Therefore try Dropout to prevent overfitting
2. Dropout:
        Dropout(p=0.1) + GELU + Normalization -> acc = 6.2 (dev) + 6.5 (test)
        -> Regularization with weight decay in Adam
3. Regularization:
        weight_decay=1e-5 -> dev acc = 0.625
        weight_decay=1e-4 -> dev acc = 0.5853211009174312
        weight_decay=1e-6 -> dev acc = 0.6110091743119266
4. Only preserved one fc layer and got the best results so far
        weight_decay=1e-6 -> dev acc = 0.629, test acc = 0.656
        weight_decay=1e-5 -> dev acc = 0.627
        weight_decay=1e-6 + dropout(p=0.1) -> dev acc = 0.631
        weight_decay=1e-5 + dropout(p=0.1) -> dev acc = 0.645
        weight_decay=1e-5 + Normalization -> dev acc = 0.59
        weight_decay=1e-5 + dropout(p=0.1) + Normalization -> dev acc = 0.59
        -> Normalization is not good for this model QAQ
5. Try CrossEntropy loss
        -> The acc increase fast but overfits fast
        lr = 2e-2 + weight_decay=1e-5 + dropout(p=0.1) -> dev acc = 0.677
        lr = 2e-2 + weight_decay=1e-5 + dropout(p=0.2) -> dev acc = 0.638
        lr = 7e-3 + weight_decay=1e-5 + dropout(p=0.1) -> dev acc = 0.675
        Current highest dev acc = 0.77

'''
import torch
import torch.nn as nn
from word_embedding import Word_Embedding
from sentence_rep import Sentence_Rep


class Model(nn.Module):
    '''
    This class that builds the BiLSTM model with a classifier.
    :param list pre_train_weight: the pre-trained weights
    :param int vocab_size: the size of vocabulary in text parser
    :param int embedding_dim: the embedding dimension (suggested 300 at least)
    :param bool from_pre_train: True if use the pre-trained wrights
    :param bool freeze: True if freeze the weights
    :param bool bow: False if builds BiLSTM
    :param int hidden_dim_bilstm: the hidden dimension of BiLSTM
    :param int hidden_layer_size: the hidden layer size of BiLSTM
    :param int num_of_classes: the number of classes, 6 or 50
    :return: a BiLSTM model with a classifier
    '''
    def __init__(self, pre_train_weight, vocab_size, embedding_dim, from_pre_train: bool, freeze: bool, bow: bool, hidden_dim_bilstm, hidden_layer_size, num_of_classes):
        super().__init__()

        self.hidden_dim_bilstm = hidden_dim_bilstm
        self.word_embedding = Word_Embedding(pre_train_weight=pre_train_weight, vocab_size=vocab_size, embedding_dim=embedding_dim, from_pre_train=from_pre_train, freeze=freeze)

        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim_bilstm, bidirectional=True, batch_first=True)

        # self.fc1 = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=hidden_layer_size)
        # self.af1 = nn.GELU()
        #
        # self.fc1_2 = nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size)
        # self.af1_2 = nn.GELU()
        #
        # self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=num_of_classes)
        # self.af2 = nn.LogSoftmax(dim=0)

        self.fc3 = nn.Linear(in_features=hidden_dim_bilstm * 2, out_features=num_of_classes)
        self.af3 = nn.LogSoftmax(dim=0)


        self.norm = nn.BatchNorm1d(2*hidden_dim_bilstm) # BatchNorm2d only accepts 4D inputs while BatchNorm1d accepts 2D or 3D inputs
        self.dropout = nn.Dropout(p=0.2) # dropout

        # self.af1 = nn.LeakyReLU(0.1)


    def forward(self, indexed_sentence):
        # ------------ word embedding -------------
        out = self.word_embedding(indexed_sentence)

        # ------------ bilstm -------------
        bilstm_out, _ = self.bilstm(out) # torch.Size([545, 20, 512])
        back = bilstm_out[:, 0, self.hidden_dim_bilstm:] # torch.Size([545, 256])
        forward = bilstm_out[:, -1, :self.hidden_dim_bilstm] # torch.Size([545, 256])
        out = torch.cat((forward, back), dim=1) # torch.Size([545, 512])

        # # ------------ Classifier -------------
        # # normalize the data
        # out = self.norm(out)
        # # Dropout randomly
        # out = self.dropout(out)
        # out = self.fc1(out)
        # out = self.af1(out)
        #
        #
        # # Dropout randomly
        # out = self.dropout(out)
        # out = self.fc1_2(out)
        # out = self.af1_2(out)
        #
        # # Dropout randomly
        # out = self.dropout(out)
        #
        # out = self.fc2(out)
        # out = self.af2(out)


        out = self.dropout(out)
        out = self.fc3(out)
        out = self.af3(out)
        return out

        # out = torch.tanh(out) # tanh, activation function 1