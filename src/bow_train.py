import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from bow import Model
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from configparser import ConfigParser
import matplotlib.pyplot as plt 
# import the configure files
config = ConfigParser()
config.read("../data/bow.config")

def plot_history(train_loss, train_accs, train_F1s, dev_loss, dev_accs, dev_F1s, num_classes, num_epoches):
    """
    Plot the figures of training and developing
    :param acc: np
    :param loss: np
    :param result_dir: path to save the figures
    """

    # generate saving path based on classes
    if num_classes == 6:
        loss_path = 'bow_Utilities/plotted_loss_bow_coase.png'
        acc_path = 'bow_Utilities/plotted_loss_bow_coase.png'
        f1_path = 'bow_Utilities/plotted_loss_bow_coase.png'
    else:
        loss_path = 'bow_Utilities/plotted_loss_bow_fine.png'
        acc_path = 'bow_Utilities/plotted_loss_bow_fine.png'
        f1_path = 'bow_Utilities/plotted_loss_bow_fine.png'

    # generate epoch list
    epochs = range(num_epoches)

    # Plot Loss
    plt.title('Training and Development Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, dev_loss, label='Development Loss')
    plt.savefig(loss_path)
    plt.close()

    # Plot Acc
    plt.title('Training and Development Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, dev_accs, label='Development Accuracy')
    plt.savefig(acc_path)
    plt.close()

    # Plot F1
    plt.title('Training and Development Macro F1')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epochs, train_F1s, label='Training macro F1')
    plt.plot(epochs, dev_F1s, label='Development macro F1')
    plt.savefig(f1_path)
    plt.close()

    # plt.plot(loss, marker='.')

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
            f1 = f1_score(output_idx, dev_labels, average="macro")
            dev_accs.append(acc)
            dev_F1s.append(f1)

    return dev_losses, dev_accs, dev_F1s


def train(t, train_data, dev_data, num_classes, pretrain, lr, epoch, batch, embedding_dim, freeze, hidden_dim, hidden_layer, pre_trained_weight=None):
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
        cnt = 0 # the number of batches
        loss_batch = 0 # the sum of average loss of each batch within a epoch
        acc_batch = 0 # the sum of average accuracy of each batch within a epoch
        f1_batch = 0 # the sum of average f1 score of each batch within a epoch

        for train_labels, train_features in iter(train_loader):
            
            train_labels = train_labels.type(torch.LongTensor) # shape (batch_size) <==> (500)
            
            # to ensure the word embedding work correctly
            if len(train_labels) != batch:
                continue

            # prediction
            output = model(train_features) # shape (batch_size, class_num) <==> (500, 50)

            # calculate loss
            loss = loss_function(output, train_labels)  # calculate loss
            loss_batch += float(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy() # shape: (64,)
        
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
                    batch, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s)
                print(
                    "Dev batch", f'epoch: {e}, loss: {dev_losses[-1]}, accuracy: {dev_accs[-1]}, f1 Score: {dev_F1s[-1]}')


        # record the data for each epoch
        train_losses.append(loss_batch/cnt)  # average loss of the batch
        train_accs.append(acc_batch/cnt)
        train_F1s.append(f1_batch/cnt)

        # print for each epoch
        print("Train epoch",
              f'epoch: {e}, loss: {loss_batch/cnt}, accuracy: {acc_batch/cnt}, f1 Score: {f1_batch/cnt}, lr: {adam_lr}')

        # update optimizer scheduler
        scheduler.step()


    # plot_history(accList, lossList "./")

    # save the model
    if num_classes == 6:
        # save training records
        np.savetxt(config.get("param","loss_bow_coase"), train_losses)
        np.savetxt(config.get("param","acc_bow_coase"), train_accs)
        np.savetxt(config.get("param","f1_bow_coase"), train_F1s)

        # save developing records
        np.savetxt(config.get("param","loss_bow_coase"), dev_losses)
        np.savetxt(config.get("param","acc_bow_coase"), dev_accs)
        np.savetxt(config.get("param","f1_bow_coase"), dev_F1s)

        #save the model
        torch.save(model, config.get("param","bow_coase_pth"))

        # plot and save
        plot_history(train_losses, train_accs, train_F1s, dev_losses, dev_accs, dev_F1s, num_classes, epoch)

        print("successfully saved the coase model!")

        
    else:
        # save training records
        np.savetxt(config.get("param","loss_bow_fine"), train_losses)
        np.savetxt(config.get("param","acc_bow_fine"), train_accs)
        np.savetxt(config.get("param","f1_bow_fine"), train_F1s)

        # save developing records
        np.savetxt(config.get("param","loss_bow_fine"), dev_losses)
        np.savetxt(config.get("param","acc_bow_fine"), dev_accs)
        np.savetxt(config.get("param","f1_bow_fine"), dev_F1s)

        # save the model
        torch.save(model, config.get("param","bow_fine_pth"))

        # plot and save
        plot_history(train_losses, train_accs, train_F1s, dev_losses, dev_accs, dev_F1s, num_classes, epoch)

        print("successfully saved the fine model!")