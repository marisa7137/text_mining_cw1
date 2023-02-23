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
config.read("bilstm.config")


def development(batch_size, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s):
    '''
    The function for model development (fine-tuning/optimisation) at the end of each epoch
    :param int batch_size: the developing batch size
    :param DataLoader dev_loader: the developing data loader
    :param Model model: the model updated in the current epoch
    :param torch.nn.NLLLoss loss_function: the loss function
    :param list dev_losses: the developing loss list
    :param list dev_accs: the developing accuracy list
    :param list dev_F1s: the developing F1 score list
    :return: dev_losses, dev_accs, dev_F1s
    '''

    # turn off gradients computation
    with torch.no_grad():
        for dev_labels, dev_features in iter(dev_loader):
            # convert the labels to tensor
            dev_labels = dev_labels.type(torch.LongTensor)  # shape (batch_size,) <==> (545,)

            # to ensure the word embedding work correctly
            if len(dev_labels) != batch_size:
                break

            # prediction
            output = model(dev_features)

            # calculate loss
            loss = loss_function(output, dev_labels)
            dev_losses.append(float(loss))

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy()  # shape: (545,)

            # calculate accuracy and f1
            acc = accuracy_score(output_idx, dev_labels)
            f1 = f1_score(output_idx, dev_labels, average="micro")
            dev_accs.append(acc)
            dev_F1s.append(f1)

    return dev_losses, dev_accs, dev_F1s




def train(t, train_data, dev_data, num_classes, pre_trained_weight=None):
    '''
    The main function for training
    :param TextParser t: test parser
    :param list train_data: the training data
    :param list dev_data: the developing data
    :param int num_classes: the number of classes, 6 or 50
    :return: None
    '''

    lr = 1e-2
    epochs = 10
    batch_size = 545

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)

    model = Model(pre_train_weight=pre_trained_weight, vocab_size=len(t.glove_vocab), embedding_dim=300, from_pre_train=True, freeze=False,
                    bow=False, hidden_dim_bilstm=256, hidden_layer_size=75, num_of_classes=num_classes)

    loss_function = torch.nn.NLLLoss(reduction='mean') # calculate the average negative log loss of a batch

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # initialize the lists that record the results
    train_losses, train_accs, train_F1s = [], [], []
    dev_losses, dev_accs, dev_F1s = [], [], []


    model.train()
    for epoch in range(epochs):
        cnt = 0 # the number of batches
        loss_batch = 0 # the sum of average loss of each batch within a epoch
        acc_batch = 0 # the sum of average accuracy of each batch within a epoch
        f1_batch = 0 # the sum of average f1 score of each batch within a epoch

        for train_labels, train_features in iter(train_loader):
            # convert the labels to tensor
            train_labels = train_labels.type(torch.LongTensor)  # shape (batch_size,) <==> (545,)

            # to ensure the word embedding work correctly
            if len(train_labels) != batch_size:
                break

            # prediction
            output = model(train_features) # shape (batch_size, class_num) <==> (545, 50)

            # calculate loss
            loss = loss_function(output, train_labels)
            loss_batch += float(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy() # shape: (545,)
            # calculate accuracy and f1
            acc = accuracy_score(output_idx, train_labels)
            f1 = f1_score(output_idx, train_labels, average="micro")
            acc_batch += acc
            f1_batch += f1

            # calculate the gradients and do back propagation
            optimizer.zero_grad()  # clean gradients
            loss.backward()  # backward pass
            optimizer.step()  # update weights

            # print for each batch
            cnt += 1
            adam_lr = optimizer.param_groups[0]['lr']
            # print("Train batch", f'epoch: {epoch}, batch: {cnt}, loss: {loss_batch}, accuracy: {acc}, f1 Score: {f1}, lr: {adam_lr}')

            # model development at the end of each epoch. There are 9 batches in each epoch
            if cnt==9:
                dev_losses, dev_accs, dev_F1s = development(batch_size, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s)
                print("Dev batch", f'epoch: {epoch}, loss: {dev_losses[-1]}, accuracy: {dev_accs[-1]}, f1 Score: {dev_F1s[-1]}')


        # record the data for each epoch
        train_losses.append(loss_batch/cnt)  # average loss of the batch
        train_accs.append(acc_batch/cnt)
        train_F1s.append(f1_batch/cnt)

        # print for each epoch
        print("Train epoch", f'epoch: {epoch}, loss: {loss_batch/cnt}, accuracy: {acc_batch/cnt}, f1 Score: {f1_batch/cnt}, lr: {adam_lr}')

        # update optimiser's scheduler
        scheduler.step()


    # plot_history(accList, lossList "./")

    # save the model
    if num_classes == 6:
        # save training records
        np.savetxt(config.get("param","loss_bilstm_coase"), train_losses)
        np.savetxt(config.get("param","acc_bilstm_coase"), train_accs)
        np.savetxt(config.get("param","f1_bilstm_coase"), train_F1s)

        # save developing records
        np.savetxt(config.get("param", "loss_bilstm_coase"), dev_losses)
        np.savetxt(config.get("param", "acc_bilstm_coase"), dev_accs)
        np.savetxt(config.get("param", "f1_bilstm_coase"), dev_F1s)

        # save the model
        torch.save(model, config.get("param","bilstm_coase_pth"))

        print("successfully saved the coarse model!")
    else:
        # save training records
        np.savetxt(config.get("param","loss_bilstm_fine"), train_losses)
        np.savetxt(config.get("param","acc_bilstm_fine"), train_accs)
        np.savetxt(config.get("param","f1_bilstm_fine"), train_F1s)
        torch.save(model, config.get("param","bilstm_fine_pth"))

        # save developing records
        np.savetxt(config.get("param", "loss_bilstm_coase"), dev_losses)
        np.savetxt(config.get("param", "acc_bilstm_coase"), dev_accs)
        np.savetxt(config.get("param", "f1_bilstm_coase"), dev_F1s)

        # save the model
        print("successfully saved the fine model!")