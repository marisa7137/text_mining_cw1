import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# def det_output(output, label, num_classes):
#     '''
#         Determine the values of TP & TN & FP & FN
#     '''
#
#     # initialization
#     TP, TN, FP, FN = 0, 0, 0, 0
#     output_vec = np.zeros(num_classes) # output vector for 6 coase classes
#     label_vec = np.zeros(num_classes)
#
#     # get the index of the class with the maximum likelihood,
#     output_idx = torch.argmax(output, dim=1).cpu().data.numpy()
#     label_idx = label
#     # and then binarize the output and label vectors
#     output_vec[output_idx] = 1
#     label_vec[label_idx] = 1
#
#     # Check one class by one class of the vectors
#     for i in range(num_classes):
#         # TP
#         if (output_vec[i]==1) and (label_vec[i]==1):
#             TP = TP + 1
#
#         # TN
#         if (output_vec[i]==0) and (label_vec[i]==0):
#             TN = TN + 1
#
#         # FP
#         if (output_vec[i]==1) and (label_vec[i]==0):
#             FP = FP + 1
#
#         # FN
#         if (output_vec[i]==0) and (label_vec[i]==1):
#             FN = FN + 1
#
#     return TP, TN, FP, FN
#
#
#
# def accuracy(output, label, num_classes):
#     '''
#         To calculate the accuracy of the output
#                     TP+TN
#         Accuracy= ---------------
#                     TP+TN+FP+FN
#     '''
#     TP, TN, FP, FN = det_output(output, label, num_classes)
#     acc = (TP+TN)/(TP+TN+FP+FN)
#     return acc
#
#
#
#
# def F1_score(output, label, num_classes):
#     '''
#         To calculate the F1 score of the output
#
#                     2 x (Precision x Recall)
#         F1 score = ----------------------------
#                         (Precision + Recall)
#
#         Prcision = TP/(TP + FP)
#         Recall = TP/(TP + FN)
#     '''
#     TP, _, FP, FN = det_output(output, label, num_classes)
#     Precision = TP/(TP + FP)
#     Recall = TP/(TP + FN)
#     f1 = (2*Precision*Recall)/(Precision + Recall)
#     return f1



def test(test_data, num_classes):
    '''
        The main function for testing
    '''

    # load the data
    test_loader = DataLoader(test_data, shuffle=True)

    # load the model
    if num_classes==6:
        model = torch.load('./biLSTM_COASE.pth')
    else:
        model = torch.load('./biLSTM_fineclass.pth')

    # evaluate model
    model = model.eval()
    
    # define the loss function and acc
    loss_function = torch.nn.NLLLoss()

    # initialization
    test_losses, test_accs, test_F1s = [], [], []

    # turn off gradients computation
    with torch.no_grad():
        for test_labels, test_features in iter(test_loader):
            test_labels = test_labels.type(torch.LongTensor)

            # predict the output
            output = model(test_features)

            # calculate the loss
            loss = loss_function(output, test_labels)
            test_losses.append(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy()  # shape: (64,)
            
            # calculate the accuracy and F1 score
            acc = accuracy_score(output_idx, test_labels)
            f1 = f1_score(output_idx, test_labels, average="micro")
            test_accs.append(acc)
            test_F1s.append(f1)

    print("Test", f'loss: {np.mean(test_losses)}, accuracy: {np.mean(test_accs)}, f1_score: {np.mean(test_F1s)}')

