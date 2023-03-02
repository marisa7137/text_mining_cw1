import numpy as np
from text_parser import TextParser
from configparser import ConfigParser

if __name__ == '__main__':
    
    config = ConfigParser()
    config.read('../data/bilstm.config')
    t_train = TextParser(pathfile=config.get("param", "path_train"), tofile=False, stopwords_pth=config.get("param", "stop_words"), fine_label_pth=config.get(
        "param", "fine_label"), coarse_label_pth=config.get("param", "coarse_label"), vocab_pth=config.get("param", "vocab"), glove_vocab_pth=config.get(
            "param", "glove_vocab_path"), glove_weight_pth=config.get("param", "glove_weight_path"))
    test = train_data = t_train.get_word_indices(
            config.get("param", "class_label"), dim=20, from_file=True)
    print(test[1])


        

