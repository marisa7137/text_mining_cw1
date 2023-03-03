import bow_train
import bow_test
import bilstm_test
import bilstm_train
from bilstm import Model
from text_parser import TextParser
import torch
import numpy as np
from configparser import ConfigParser
import argparse

if __name__ == '__main__':
    """
    The main function of this script loads configuration settings from a specified file, trains or tests a BiLSTM model on text data, and outputs results.

    Required arguments:
    --config (str): Path to the configuration file that specifies training and testing parameters.

    Optional arguments:
    --train: If included, runs the model in training mode and saves the resulting model.
    --test: If included, runs the model in testing mode, requiring a saved model to be loaded.

    Returns:
    None.
    """

    torch.manual_seed(1)
    np.random.seed(1)
    config = ConfigParser()
    parser = argparse.ArgumentParser(
        description='Argument parser for loading config, training, testing')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file', default="../data/bilstm.config")
    parser.add_argument('--train', action='store_true',
                        help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true',
                        help='Testing mode - needs a model to load')
    # Receive the args
    args = parser.parse_args()
    # Read the parser
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

    # Check wether it is pretrain or not
    if (config.getboolean("param", "pretrain")):
        print("Pretrain Option: On")
        train_data = t_train.get_word_indices_from_glove(
            config.get("param", "class_label"), dim=20)
        dev_data = t_dev.get_word_indices_from_glove(
            config.get("param", "class_label"), dim=20)
        test_data = t_test.get_word_indices_from_glove(
            config.get("param", "class_label"), dim=20)
        pre_trained_weight = t_train.glove_embedding
    else:
        print("Pretrain Option: Off")
        train_data = t_train.get_word_indices(
            config.get("param", "class_label"), dim=20, from_file=True)
        dev_data = t_dev.get_word_indices(
            config.get("param", "class_label"), dim=20, from_file=True)
        test_data = t_test.get_word_indices(
            config.get("param", "class_label"), dim=20, from_file=True)
        pre_trained_weight = None

    # Training Function
    if(args.train):
        print("Mode: Training")
        if(config.get("param", "class_label") == "fine"):
            print("Class Label: fine")
            # do the train function
            if(config.get("param", "model") == "bow"):
                bow_train.train(t_train, train_data, dev_data,
                                num_classes=50,
                                pretrain=config.getboolean(
                                    "param", "pretrain"),
                                lr=config.getfloat("param", "lr"),
                                epoch=config.getint("param", "epoch"),
                                embedding_dim=config.getint(
                                    "param", "embedding_dim"),
                                batch=config.getint("param", "batch"),
                                freeze=config.getboolean("param", "freeze"),
                                hidden_dim=config.getint(
                                    "param", "hidden_dim"),
                                hidden_layer=config.getint(
                                    "param", "hidden_layer"),
                                output_file= config.get(
                                       "param", "dev_fine_output"),
                                pre_trained_weight=pre_trained_weight)
            elif (config.get("param", "model") == "bilstm"):
                print("Model: bilstm")
                print("----------------------------------------")
                bilstm_train.train(t_train, train_data, dev_data,
                                   num_classes=50,
                                   pretrain=config.getboolean(
                                       "param", "pretrain"),
                                   lr=config.getfloat("param", "lr"),
                                   epoch=config.getint("param", "epoch"),
                                   embedding_dim=config.getint(
                                       "param", "embedding_dim"),
                                   batch=config.getint("param", "batch"),
                                   freeze=config.getboolean("param", "freeze"),
                                   hidden_dim=config.getint(
                                       "param", "hidden_dim"),
                                   hidden_layer=config.getint(
                                       "param", "hidden_layer"),
                                   output_file= config.get(
                                       "param", "dev_fine_output"),
                                   pre_trained_weight=pre_trained_weight)
        elif(config.get("param", "class_label") == "coarse"):
            print("Class Label: coarse")
            if(config.get("param", "model") == "bow"):
                bow_train.train(t_train, train_data, dev_data,
                                num_classes=6,
                                pretrain=config.getboolean(
                                    "param", "pretrain"),
                                lr=config.getfloat("param", "lr"),
                                epoch=config.getint("param", "epoch"),
                                embedding_dim=config.getint(
                                    "param", "embedding_dim"),
                                batch=config.getint("param", "batch"),
                                freeze=config.getboolean("param", "freeze"),
                                hidden_dim=config.getint(
                                    "param", "hidden_dim"),
                                hidden_layer=config.getint(
                                    "param", "hidden_layer"),
                                output_file= config.get(
                                       "param", "dev_fine_output"),
                                pre_trained_weight=pre_trained_weight)
            elif (config.get("param", "model") == "bilstm"):
                print("Model: bilstm")
                bilstm_train.train(t_train, train_data, dev_data,
                                   num_classes=6,
                                   pretrain=config.getboolean(
                                       "param", "pretrain"),
                                   lr=config.getfloat("param", "lr"),
                                   epoch=config.getint("param", "epoch"),
                                   embedding_dim=config.getint(
                                       "param", "embedding_dim"),
                                   batch=config.getint("param", "batch"),
                                   freeze=config.getboolean("param", "freeze"),
                                   hidden_dim=config.getint(
                                       "param", "hidden_dim"),
                                   hidden_layer=config.getint(
                                       "param", "hidden_layer"),
                                   output_file=config.get(
                                       "param", "dev_coarse_output"),
                                   pre_trained_weight=pre_trained_weight)
    # Tesing Fucntion
    if(args.test):
        print("Mode: Testing")
        if(config.get("param", "class_label") == "fine"):
            print("Class Label: fine")
            if(config.get("param", "model") == "bow"):
                bow_test.test(t_test, test_data, num_classes=50, model_pth=config.get(
                    "param", "bow_fine_pth"), output_pth=config.get("param", "fine_output"))
            elif (config.get("param", "model") == "bilstm"):
                bilstm_test.test(t_test, test_data, num_classes=50, model_pth=config.get(
                    "param", "bilstm_fine_pth"), output_pth=config.get("param", "fine_output"))

        elif(config.get("param", "class_label") == "coarse"):
            print("Class Label: coarse")
            if(config.get("param", "model") == "bow"):
                print("Model: bow")
                bow_test.test(t_test, test_data, num_classes=6, model_pth=config.get(
                    "param", "bow_coase_pth"), output_pth=config.get("param", "coarse_output"))
            elif (config.get("param", "model") == "bilstm"):
                print("Model: bilstm")
                bilstm_test.test(t_test, test_data, num_classes=6, model_pth=config.get(
                    "param", "bilstm_coase_pth"), output_pth=config.get("param", "coarse_output"))
