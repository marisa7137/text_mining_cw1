import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def test_output(t, predValue_idx, num_classes, output_pth):
    """
        Function:
            Generate the format output text files including the raw sentecnce, groudtruth label and predicted laebl
        punctions
        Args:
            t: text parser object
            predValue_idx: the index value from the model predit output
            num_classes: define the fine or coarse class
            output_pth: the path for storing the output
    """

    if (num_classes == 6):
        raw_sentence = t.raw_sentences
        groundTruth = t.coarse_pair
        label_mapping = t.coarse_labels

    else:
        raw_sentence = t.raw_sentences
        groundTruth = t.fine_pair
        label_mapping = t.fine_labels

    with open(output_pth, "w") as f:
        column_width = 20
        # Create table header
        table_header = "{:<{}} {:<{}} {:<{}}\n".format(
            "Groud Truth Label", column_width, "Predict Label ", column_width, "Question", column_width)
        table_rows = ""

        for idx in range(0, len(predValue_idx)):
            raw_sentence_single = raw_sentence[idx]
            groundTruth_single = groundTruth[idx][0]

            predValue_idx_single = predValue_idx[idx]

            # mapping: idex -> label
            predLabel_single = label_mapping[predValue_idx_single]
            table_row = "{:<{}} {:<{}} {:<{}}\n".format(
                groundTruth_single, column_width, predLabel_single, column_width, raw_sentence_single, column_width)
            table_rows += table_row
            table = table_header + table_rows
        f.writelines(table)


def test(t,test_data, num_classes, model_pth,output_pth):
    '''
    The main function for testing
    :param list test_data: the test data
    :param int num_classes: the number of classes, 6 or 50
    :param str model_pth: the path of trained model file
    :return: None
    '''

    # load the data
    batch_size = 500
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    

    # load the model
    if num_classes==6:
        model = torch.load(model_pth)
    else:
        model = torch.load(model_pth)

    # evaluate model
    model = model.eval()

    # define the loss function
    loss_function = torch.nn.NLLLoss(reduction='mean') # calculate the average negative log loss of a batch

    # initialization
    test_losses, test_accs, test_F1s = [], [], []
    predValue_idx = []

    # turn off gradients computation
    with torch.no_grad():
        for test_labels, test_features in iter(test_loader):
            test_labels = test_labels.type(torch.LongTensor) # shape (545, )

            # to ensure the word embedding work correctly
            if len(test_labels) != batch_size:
                break
            
            # predict the output
            output = model(test_features)

            # calculate the loss
            loss = loss_function(output, test_labels)

            test_losses.append(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy()  # shape: (1,)
            predValue_idx = output_idx
            #print(test_labels.shape)
            
            # calculate the accuracy and F1 score
            acc = accuracy_score(output_idx, test_labels)
            f1 = f1_score(output_idx, test_labels, average="macro")
            test_accs.append(acc)
            test_F1s.append(f1)
            #print(acc, "acc")
    
                
    #print("Test", f'loss: {np.mean(test_losses)}, accuracy: {np.mean(test_accs)}, f1_score: {np.mean(test_F1s)}')
    print(
        "Test", f'loss: {np.mean(test_losses)}, accuracy: {np.mean(test_accs)}, f1_score: {np.mean(test_F1s)}')
    test_output(t, predValue_idx, num_classes, output_pth)

