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
# Added the random seed generator


# python src\question_classifier.py --train --config "src\bilstm.config" --class_label "fine"

if __name__ == '__main__':
    torch.manual_seed(6666666)
    np.random.seed(6)
    config = ConfigParser()
    parser = argparse.ArgumentParser(description='Argument parser for loading config, training, testing')
    parser.add_argument('--config', type=str, required=True, help='Configuration file',default="src/bilstm.config")
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
    parser.add_argument('--class_label',type=str,required=True, help= 'different class fine',default= "fine")

    args = parser.parse_args()
    config.read(args.config)

    t_train = TextParser(pathfile=config.get("param","path_train"))
    train_data = t_train.get_word_indices(args.class_label, dim=20, from_file=True)
    t_test = TextParser(pathfile=config.get("param","path_dev"))
    test_data = t_test.get_word_indices(args.class_label, dim=20, from_file=True)

    if(args.train):
        if(args.class_label == "fine"):
             # do the train function
            if(config.get("param","model")=="bow"):
                pass
            elif (config.get("param","model")=="bilstm"):
                bilstm_train.train(t_train, train_data, num_classes=50)
        elif(args.class_label == "coarse"):
            if(config.get("param","model")=="bow"):
                pass
            elif (config.get("param","model")=="bilstm"):
                bilstm_train.train(t_train, train_data, num_classes=6)
            
            
    if(args.test):
        if(args.class_label == "fine"):
            if(config.get("param","model")=="bow"):
                pass
            elif (config.get("param","model")=="bilstm"):
                bilstm_test.test(test_data, num_classes=50)
        elif(args.class_label == "coarse"):
            if(config.get("param","model")=="bow"):
                pass
            elif (config.get("param","model")=="bilstm"):
                bilstm_test.test(test_data, num_classes=6)
            
        
