'''
    ***** REMINDER *****

    ** Before submitting:
    1. Remove unused/impermissible libraries
    2. Remove prints that are not required. e.g. the prints of saving in the training file
    3. Check the settings are as required. e.g. epoch=10


    ** Execution:
    python src/question_classifier.py --train --config "src/bilstm.config" --class_label "fine"
    python src/question_classifier.py --test --config "src/bilstm.config" --class_label "fine"
'''

import argparse
from configparser import ConfigParser
import numpy as np
import torch
from text_parser import TextParser
from bilstm import Model
import bilstm_train
import bilstm_test
import numpy as np
import torch
from text_parser import TextParser
from configparser import ConfigParser
import BagOfWords_test
import BagofWords_train
# Added the random seed generator


if __name__ == '__main__':
    torch.manual_seed(6)
    np.random.seed(6)
    config = ConfigParser()
    parser = argparse.ArgumentParser(description='Argument parser for loading config, training, testing')
    parser.add_argument('--config', type=str, required=True, help='Configuration file',default="src/bilstm.config")
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
    parser.add_argument('--class_label',type=str,required=True, help= 'different class fine',default= "fine")

    args = parser.parse_args()
    config.read(args.config)

    t_train = TextParser(pathfile=config.get("param","path_train"),tofile=False)
    train_data = t_train.get_word_indices(args.class_label, dim=20, from_file=True)
    t_dev = TextParser(pathfile=config.get("param","path_dev"),tofile=False)
    dev_data = t_dev.get_word_indices(args.class_label, dim=20, from_file=True) # development data (validation)

    # test data have not been not read yet
    test_data = []

    if(args.train):
        if(args.class_label == "fine"):
             # do the train function
            if(config.get("param","model")=="bow"):
                BagofWords_train.train(t_train, train_data, dev_data, num_classes=50)
            elif (config.get("param","model")=="bilstm"):
                bilstm_train.train(t_train, train_data, dev_data, num_classes=50)
        elif(args.class_label == "coarse"):
            if(config.get("param","model")=="bow"):
                BagofWords_train.train(t_train, train_data, dev_data, num_classes=6)
            elif (config.get("param","model")=="bilstm"):
                bilstm_train.train(t_train, train_data, dev_data, num_classes=6)
            
            
    if(args.test):
        if(args.class_label == "fine"):
            if(config.get("param","model")=="bow"):
                BagOfWords_test.test(test_data, num_classes=50)
            elif (config.get("param","model")=="bilstm"):
                bilstm_test.test(test_data, num_classes=50,model_pth=config.get("param","bilstm_fine_pth"))
        elif(args.class_label == "coarse"):
            if(config.get("param","model")=="bow"):
                BagOfWords_test.test(test_data, num_classes=6)
            elif (config.get("param","model")=="bilstm"):
                bilstm_test.test(test_data, num_classes=6,model_pth=config.get("param","bilstm_coase_pth"))
            
        
