import torch
import torch.nn as nn


# Create a bidirectional LSTM
class BILSTM(nn.Module):
    def __init__(self, input_size=10, num_classes=50, num_layers=2, hidden_size = 256):
        super(BILSTM, self).__init__()
        # initialize
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # set bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fully_connect = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # put the input into biLSTM
        # output_lstm (L, N, D*H_out)
        # L: sequence length
        # N: batch size
        # D*H_out: 2 (due to biLSTM) * hidden_size
        output_lstm, _, _ = self.lstm(x) 
        output_lstm = output_lstm[-1, :, :]

        # Classification process
        output = self.fully_connect(output_lstm)

        return output

