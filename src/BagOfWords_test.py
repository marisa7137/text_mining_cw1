import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/BagofWords.config")

def test(test_data, num_classes):
    '''
        The main function for testing
    '''

    # load the data
    
    batch_size = 545
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    

    # load the model
    if num_classes==6:
        model = torch.load(config.get("param","bow_coase_pth"))
    else:
        model = torch.load(config.get("param","bow_fine_pth"))

    # evaluate model
    model = model.eval()

    # define the loss function
    loss_function = torch.nn.NLLLoss(reduction='mean') # calculate the average negative log loss of a batch

    # initialization
    test_losses, test_accs, test_F1s = [], [], []

    # turn off gradients computation
    with torch.no_grad():
        for test_labels, test_features in iter(test_loader):
            test_labels = test_labels.type(torch.LongTensor) # shape (1)

            # predict the output
            output = model(test_features)

            # calculate the loss
            loss = loss_function(output, test_labels)

            test_losses.append(loss)

            # get the index of the class with the maximum likelihood
            output_idx = torch.argmax(output, dim=1).cpu().data.numpy()  # shape: (1,)
            print(test_labels.shape)
            
            # calculate the accuracy and F1 score
            acc = accuracy_score(output_idx, test_labels)
            f1 = f1_score(output_idx, test_labels, average="micro")
            test_accs.append(acc)
            test_F1s.append(f1)
            print(acc, "acc")
    
                
                
                



    print("Test", f'loss: {np.mean(test_losses)}, accuracy: {np.mean(test_accs)}, f1_score: {np.mean(test_F1s)}')

