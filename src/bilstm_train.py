import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bilstm import Model
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/bilstm.config")




def train(t, train_data, num_classes):
    '''
            The main function for testing
    '''

    lr = 7e-2
    epochs = 10
    batch_size = 545

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = Model(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=300, from_pre_train=False, freeze=False,
                    bow=False, hidden_dim_bilstm=20, hidden_layer_size=30, num_of_classes=num_classes)

    loss_function = torch.nn.NLLLoss(reduction='mean') # calculate the average negative log loss of a batch

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

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

        # update optimizer scheduler
        scheduler.step()


    # plot_history(accList, lossList "./")

    # save the model
    if num_classes == 6:
        np.savetxt(config.get("param","loss_bilstm_coase"), losses)
        np.savetxt(config.get("param","acc_bilstm_coase"), train_accs)
        np.savetxt(config.get("param","f1_bilstm_coase"), train_F1s)
        torch.save(model, config.get("param","bilstm_coase_pth"))
        print("successfully saved the coase model!")
    else:
        np.savetxt(config.get("param","loss_bilstm_fine"), losses)
        np.savetxt(config.get("param","acc_bilstm_fine"), train_accs)
        np.savetxt(config.get("param","f1_bilstm_fine"), train_F1s)
        torch.save(model, config.get("param","bilstm_fine_pth"))
        print("successfully saved the fine model!")