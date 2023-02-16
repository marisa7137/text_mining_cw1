import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score




def train(t, train_data):
    '''
            The main function for testing
    '''

    lr = 0.001
    epochs = 10
    batch_size = 500
    num_classes = len(t.labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    model = Model(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=20, from_pre_train=False, freeze=False,
                    bow=False, hidden_dim_bilstm=20, hidden_layer_size=30, num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.NLLLoss()
    losses, train_accs, train_F1s = [], [], []
    model.train()

    for epoch in range(epochs):
        for train_labels, train_features in iter(train_loader):
            train_labels = train_labels.type(torch.LongTensor)
            if len(train_labels) != batch_size:
                continue
            output = model(train_features)

            # compute loss
            loss = loss_function(output, train_labels)  # compute los
            optimizer.zero_grad()  # clean gradients
            loss.backward()  # backward pass
            optimizer.step()  # update weights
            losses.append(float(loss) / batch_size)  # average loss of the batch

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy() # shape: (64,)

            # accuracy and f1
            acc = accuracy_score(output_idx, train_labels)
            f1 = f1_score(output_idx, train_labels, average="micro")
            train_accs.append(acc)
            train_F1s.append(f1)


        print("Train", f'epoch: {epoch}, loss: {float(loss) / batch_size}, accuracy: {acc}, f1 Score: {f1}, lr: {lr}')

    np.savetxt("loss_biLSTM.txt", losses)
    np.savetxt("acc_biLSTM.txt", train_accs)

    # plot_history(accList, lossList "./")

    # save the model
    if num_classes == 6:
        torch.save(model, './biLSTM_COASE.pth')
    else:
        torch.save(model, './biLSTM_fineclass.pth')










# #input dim 1x10
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
#
#
# # 不许用tqdm这个library到时候要改！！！！！！！！！！！！！
#
#
# from torch.utils.data import (DataLoader)
# from biLSTM import BILSTM
#
#
# ## hyperparameters settings
# # for the model
# num_layers=2
# hidden_size = 128
# # for training
# learning_rate = 1e-3
# batch_size = 50
# epochs = 20
#
# # load the training data
# train_dataset = ""
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# # load the model
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# model = BILSTM(num_layers=num_layers, hidden_size = hidden_size)
#
# # Loss and optimizer
# loss_f = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # training
# for e in range(epochs):
#     for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
#         # Preprocess the data
#         inputs = inputs.to(device=device)
#         inputs = inputs.squeeze(1)
#         labels = labels.to(device=device)
#
#         # forward propagation
#         scores = model(inputs)
#         loss = loss_f(scores, labels)
#
#         # backward propagation
#         optimizer.zero_grad()
#         loss.backward()
#
#         # gradient descent or adam step
#         optimizer.step()