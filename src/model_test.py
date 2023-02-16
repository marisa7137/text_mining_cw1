import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

def det_output(output, label, class_num):
    '''
        Determine the values of TP & TN & FP & FN
    '''

    # initialization
    TP, TN, FP, FN = 0, 0, 0, 0
    output_vec = np.zeros(class_num) # output vector for 6 coase classes
    label_vec = np.zeros(class_num)

    # get the index of the class with the maximum likelihood,
    output_idx = torch.argmax(output, dim=1).cpu().data.numpy()
    label_idx = label
    # and then binarize the output and label vectors
    output_vec[output_idx] = 1
    label_vec[label_idx] = 1

    # Check one class by one class of the vectors
    for i in range(class_num):
        # TP
        if (output_vec[i]==1) and (label_vec[i]==1):
            TP = TP + 1

        # TN
        if (output_vec[i]==0) and (label_vec[i]==0):
            TN = TN + 1

        # FP
        if (output_vec[i]==1) and (label_vec[i]==0):
            FP = FP + 1

        # FN
        if (output_vec[i]==0) and (label_vec[i]==1):
            FN = FN + 1
    
    return TP, TN, FP, FN



def accuracy(output, label, class_num):
    '''
        To calculate the accuracy of the output
                    TP+TN
        Accuracy= ---------------
                    TP+TN+FP+FN
    '''
    TP, TN, FP, FN = det_output(output, label, class_num)
    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc




def F1_score(output, label, class_num):
    '''
        To calculate the F1 score of the output

                    2 x (Precision x Recall) 
        F1 score = ----------------------------
                        (Precision + Recall)
    
        Prcision = TP/(TP + FP)
        Recall = TP/(TP + FN)
    '''
    TP, _, FP, FN = det_output(output, label, class_num)
    Precision = TP/(TP + FP)
    Recall = TP/(TP + FN)
    f1 = (2*Precision*Recall)/(Precision + Recall)
    return f1



def main():
    '''
        The main function for testing
    '''
    # COASE or Fine Class
    class_num = 6

    # load the data
    test_loader = [] # combine this with testscript.py?

    # load the model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('./biLSTM.pth') 
    
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
            
            # calculate the accuracy and F1 score
            acc = accuracy(output, test_labels, class_num)
            f1 = F1_score(output,test_labels, class_num)
            test_accs.append(acc)
            test_F1s.append(f1)
    print("Test", f'loss: {np.mean(test_losses)}, accuracy: {np.mean(test_accs)}, f1_score: {np.mean(test_F1s)}')


if __name__ == '__main__':
    main()