#input dim 1x10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 


# 不许用tqdm这个library到时候要改！！！！！！！！！！！！！


from torch.utils.data import (DataLoader)
from biLSTM import BILSTM


## hyperparameters settings
# for the model
num_layers=2
hidden_size = 128
# for training
learning_rate = 1e-3
batch_size = 50
epochs = 20

# load the training data
train_dataset = ""
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
model = BILSTM(num_layers=num_layers, hidden_size = hidden_size)

# Loss and optimizer
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training
for e in range(epochs):
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        # Preprocess the data
        inputs = inputs.to(device=device)
        inputs = inputs.squeeze(1)
        labels = labels.to(device=device)

        # forward propagation
        scores = model(inputs)
        loss = loss_f(scores, labels)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()