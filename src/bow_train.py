import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bow import Model
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from configparser import ConfigParser
from bow_test import test_output
# import the configure files
config = ConfigParser()
config.read("../data/bow.config")


def development(batch_size, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s,t,numofclasses,output_path):
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
            # shape (batch_size,) <==> (545,)
            dev_labels = dev_labels.type(torch.LongTensor)

            # to ensure the word embedding work correctly
            if len(dev_labels) != batch_size:
                break

            # prediction
            output = model(dev_features)

            # calculate loss
            loss = loss_function(output, dev_labels)
            dev_losses.append(float(loss))

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(
                output, dim=1).cpu().data.numpy()  # shape: (545,)

            # calculate accuracy and f1
            acc = accuracy_score(output_idx, dev_labels)
            f1 = f1_score(output_idx, dev_labels, average="macro")
            dev_accs.append(acc)
            dev_F1s.append(f1)
            test_output(t, output_idx, numofclasses,output_path)
            with open(output_path, "r+") as f:
                old_content = f.read()  # Read existing content
                f.seek(0)  # Move cursor to beginning of file
                f.write("The macro F1 for this experiment is {}\n ".format(dev_F1s[-1]) + old_content)  # Write new content and old content back to file

    return dev_losses, dev_accs, dev_F1s


def train(t, train_data, dev_data, num_classes, pretrain, lr, epoch, batch, embedding_dim, freeze, hidden_dim, hidden_layer,output_file,pre_trained_weight=None):
    '''
    The main function for training
    :param TextParser t: test parser
    :param list train_data: the training data
    :param list dev_data: the developing data
    :param int num_classes: the number of classes, 6 or 50
    :return: None
    '''

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch, shuffle=True)

    if(pretrain):
        model = Model(pre_train_weight=pre_trained_weight, vocab_size=len(t.glove_vocab), embedding_dim=embedding_dim, from_pre_train=True, freeze=freeze,
                      bow=True, hidden_dim_bilstm=hidden_dim, hidden_layer_size=hidden_layer, num_of_classes=num_classes)
    else:
        model = Model(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=embedding_dim, from_pre_train=False, freeze=freeze,
                      bow=True, hidden_dim_bilstm=hidden_dim, hidden_layer_size=hidden_layer, num_of_classes=num_classes)

    # calculate the average negative log loss of a batch
    loss_function = torch.nn.NLLLoss(reduction='mean')
    loss_function = nn.CrossEntropyLoss()

    # L2 Regularization with weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # initialize the lists that record the results
    train_losses, train_accs, train_F1s = [], [], []
    dev_losses, dev_accs, dev_F1s = [], [], []

    model.train()
    for e in range(epoch):
        cnt = 0  # the number of batches
        loss_batch = 0  # the sum of average loss of each batch within a epoch
        acc_batch = 0  # the sum of average accuracy of each batch within a epoch
        f1_batch = 0  # the sum of average f1 score of each batch within a epoch

        for train_labels, train_features in iter(train_loader):

            # shape (batch_size) <==> (500)
            train_labels = train_labels.type(torch.LongTensor)

            # to ensure the word embedding work correctly
            if len(train_labels) != batch:
                continue

            # prediction
            # shape (batch_size, class_num) <==> (500, 50)
            output = model(train_features)

            # calculate loss
            loss = loss_function(output, train_labels)  # calculate loss
            loss_batch += float(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(
                output, dim=1).cpu().data.numpy()  # shape: (64,)

            # accuracy and f1
            acc = accuracy_score(output_idx, train_labels)
            f1 = f1_score(output_idx, train_labels, average="macro")
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
            if cnt == 9:
                dev_losses, dev_accs, dev_F1s = development(
                    batch, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s,t,num_classes,output_file)
                print(
                    "Dev batch", f'epoch: {e+1}, loss: {dev_losses[-1]}, accuracy: {dev_accs[-1]}, f1 Score: {dev_F1s[-1]}')

        # record the data for each epoch
        train_losses.append(loss_batch/cnt)  # average loss of the batch
        train_accs.append(acc_batch/cnt)
        train_F1s.append(f1_batch/cnt)

        # print for each epoch
        print("Train epoch",
              f'epoch: {e+1}, loss: {loss_batch/cnt}, accuracy: {acc_batch/cnt}, f1 Score: {f1_batch/cnt}, lr: {adam_lr}')

        # update optimizer scheduler
        scheduler.step()

    # save the model
    if num_classes == 6:

        # save the model
        torch.save(model, config.get("param", "bow_coase_pth"))
        print("successfully saved bow coarse model!")

    else:

        # save the model
        torch.save(model, config.get("param", "bow_fine_pth"))

        print("successfully saved the bow fine model!")
