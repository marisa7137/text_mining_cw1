'''
    ***** REMINDER *****
    ** Before submitting:
    1. Remove unused/impermissible libraries
    2. Remove prints that are not required. e.g. the prints of saving in the training file
    3. Check the settings are as required. e.g. epoch=10
    4. Maybe remove the BILSTM & BOW utilities file to fit the requirement of the structure
    5. Check if there is a README file
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
from pathlib import Path
# Added the random seed generator


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    config = ConfigParser()
    parser = argparse.ArgumentParser(
        description='Argument parser for loading config, training, testing')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file', default="src/bilstm.config")
    parser.add_argument('--train', action='store_true',
                        help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true',
                        help='Testing mode - needs a model to load')
    parser.add_argument('--class_label', type=str, required=True,
                        help='different class fine', default="fine")
    parser.add_argument('--pretrain', type=bool, required=False,
                        help='Wether or not using the pretrain', default=False)

    args = parser.parse_args()
    config.read(args.config)

    # training data
    t_train = TextParser(pathfile=config.get("param", "path_train"), tofile=False, stopwords_pth=config.get("param", "stop_words"), fine_label_pth=config.get(
        "param", "fine_label"), coarse_label_pth=config.get("param", "coarse_label"), vocab_pth=config.get("param", "vocab"), glove_vocab_pth=config.get(
            "param", "glove_vocab_path"), glove_weight_pth=config.get("param", "glove_weight_path"))
    # developemnt data
    t_dev = TextParser(pathfile=config.get("param", "path_dev"), tofile=False, stopwords_pth=config.get("param", "stop_words"), fine_label_pth=config.get(
        "param", "fine_label"), coarse_label_pth=config.get("param", "coarse_label"), vocab_pth=config.get("param", "vocab"), glove_vocab_pth=config.get(
            "param", "glove_vocab_path"), glove_weight_pth=config.get("param", "glove_weight_path"))
    # test data
    t_test = TextParser(pathfile=config.get("param", "path_test"), tofile=False, stopwords_pth=config.get("param", "stop_words"), fine_label_pth=config.get(
        "param", "fine_label"), coarse_label_pth=config.get("param", "coarse_label"), vocab_pth=config.get("param", "vocab"), glove_vocab_pth=config.get(
            "param", "glove_vocab_path"), glove_weight_pth=config.get("param", "glove_weight_path"))

    if (args.pretrain):
        print(args.pretrain)
        print("zhixing pretrain")
        train_data = t_train.get_word_indices_from_glove(
            args.class_label, dim=20)
        dev_data = t_dev.get_word_indices_from_glove(args.class_label, dim=20)
        test_data = t_test.get_word_indices_from_glove(
            args.class_label, dim=20)
        pre_trained_weight = t_train.glove_embedding

    else:
        print("no pretrain")
        train_data = t_train.get_word_indices(
            args.class_label, dim=20, from_file=True)
        dev_data = t_dev.get_word_indices(
            args.class_label, dim=20, from_file=True)
        test_data = t_test.get_word_indices(
            args.class_label, dim=20, from_file=True)
        pre_trained_weight = None

    if(args.train):
        if(args.class_label == "fine"):
            # do the train function
            if(config.get("param", "model") == "bow"):
                BagofWords_train.train(t_train, train_data, dev_data, 
                                       num_classes=50,
                                       pretrain=args.pretrain,
                                       lr=config.getfloat("param", "lr"),
                                       epoch=config.getint("param", "epoch"),
                                       embedding_dim=config.getint(
                                            "param", "embedding_dim"),
                                       batch=config.getint("param", "batch"),
                                       hidden_dim=config.getint(
                                            "param", "hidden_dim"),
                                       hidden_layer=config.getint(
                                            "param", "hidden_layer"),
                                        pre_trained_weight=pre_trained_weight)
            elif (config.get("param", "model") == "bilstm"):
                bilstm_train.train(t_train, train_data, dev_data,
                                   num_classes = 50,
                                   pretrain=args.pretrain,
                                   lr=config.getfloat("param", "lr"),
                                   epoch=config.getint("param", "epoch"),
                                   embedding_dim=config.getint(
                                       "param", "embedding_dim"),
                                   batch=config.getint("param", "batch"),
                                   hidden_dim=config.getint(
                                       "param", "hidden_dim"),
                                   hidden_layer=config.getint(
                                       "param", "hidden_layer"),
                                   pre_trained_weight=pre_trained_weight)
        elif(args.class_label == "coarse"):
            if(config.get("param", "model") == "bow"):
                BagofWords_train.train(t_train, train_data, dev_data, 
                                       num_classes=6,
                                       pretrain=args.pretrain,
                                       lr=config.getfloat("param", "lr"),
                                       epoch=config.getint("param", "epoch"),
                                       embedding_dim=config.getint(
                                            "param", "embedding_dim"),
                                       batch=config.getint("param", "batch"),
                                       hidden_dim=config.getint(
                                            "param", "hidden_dim"),
                                       hidden_layer=config.getint(
                                            "param", "hidden_layer"),
                                       pre_trained_weight=pre_trained_weight)
            elif (config.get("param", "model") == "bilstm"):
                bilstm_train.train(t_train, train_data, dev_data,
                                   num_classes = 6,
                                   pretrain=args.pretrain,
                                   lr=config.getfloat("param", "lr"),
                                   epoch=config.getint("param", "epoch"),
                                   embedding_dim=config.getint(
                                       "param", "embedding_dim"),
                                   batch=config.getint("param", "batch"),
                                   hidden_dim=config.getint(
                                       "param", "hidden_dim"),
                                   hidden_layer=config.getint(
                                       "param", "hidden_layer"),
                                   pre_trained_weight=pre_trained_weight)
                

    if(args.test):
        if(args.class_label == "fine"):
            if(config.get("param", "model") == "bow"):
                BagOfWords_test.test(test_data, num_classes=50, model_pth=config.get(
                    "param","bow_fine_pth"))
            elif (config.get("param", "model") == "bilstm"):
                bilstm_test.test(test_data, num_classes=50, model_pth=config.get(
                    "param", "bilstm_fine_pth"))
        elif(args.class_label == "coarse"):
            if(config.get("param", "model") == "bow"):
                BagOfWords_test.test(test_data, num_classes=6, model_pth=config.get(
                    "param","bow_coase_pth"))
            elif (config.get("param", "model") == "bilstm"):
                bilstm_test.test(test_data, num_classes=6, model_pth=config.get(
                    "param", "bilstm_coase_pth"))
