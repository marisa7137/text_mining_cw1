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

    lr = 1e-1
    epochs = 10
    batch_size = 500
    num_classes = len(t.fine_labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    model = Model(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=20, from_pre_train=False, freeze=False,
                    bow=False, hidden_dim_bilstm=20, hidden_layer_size=30, num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.NLLLoss(reduction='mean') # calculate the average negative log loss of a batch
    losses, train_accs, train_F1s = [], [], []
    model.train()

    for epoch in range(epochs):
        cnt = 0 # the number of batches
        loss_batch = 0 # the sum of average loss of each batch within a epoch
        acc_batch = 0 # the sum of average accuracy of each batch within a epoch
        f1_batch = 0 # the sum of average f1 score of each batch within a epoch

        for train_labels, train_features in iter(train_loader):
            train_labels = train_labels.type(torch.LongTensor) # shape (batch_size) <==> (500)
            if len(train_labels) != batch_size:
                continue
            output = model(train_features) # shape (batch_size, class_num) <==> (500, 50)

            # compute loss
            loss = loss_function(output, train_labels)  # calculate loss
            loss_batch += float(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy() # shape: (64,)

            # accuracy and f1
            acc = accuracy_score(output_idx, train_labels)
            f1 = f1_score(output_idx, train_labels, average="micro")
            acc_batch += acc
            f1_batch += f1

            # network propagation
            optimizer.zero_grad()  # clean gradients
            loss.backward()  # backward pass
            optimizer.step()  # update weights

            # print for each batch
            cnt += 1
            adam_lr = optimizer.param_groups[0]['lr']
            # print("Train batch", f'epoch: {epoch}, batch: {cnt}, loss: {loss_batch}, accuracy: {acc}, f1 Score: {f1}, lr: {adam_lr}')

        # record the data for each epoch
        losses.append(loss_batch/cnt)  # average loss of the batch
        train_accs.append(acc_batch/cnt)
        train_F1s.append(f1_batch/cnt)
        # print for each epoch
        print("Train epoch", f'epoch: {epoch}, loss: {loss_batch/cnt}, accuracy: {acc_batch/cnt}, f1 Score: {f1_batch/cnt}, lr: {adam_lr}')



    # plot_history(accList, lossList "./")

    # save the model
    if num_classes == 6:
        np.savetxt("loss_biLSTM_COASE.txt", losses)
        np.savetxt("acc_biLSTM_COASE.txt", train_accs)
        np.savetxt("f1_biLSTM_COASE.txt", train_F1s)
        torch.save(model, './biLSTM_COASE.pth')
        print("successfully saved the model!")
    else:
        np.savetxt("loss_biLSTM_fineclass.txt", losses)
        np.savetxt("acc_biLSTM_fineclass.txt", train_accs)
        np.savetxt("f1_biLSTM_fineclass.txt", train_F1s)
        torch.save(model, './biLSTM_fineclass.pth')
        print("successfully saved the model!")










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