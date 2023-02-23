import random
import os
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/bilstm.config")

random.seed(6)

def random_split(input_data, shuffle=True, ratio=0):
    '''
    Function: 
        Randomly split the input file into two seperate files with a predefined ratio
    Args:
        input_data: total data
        shuffle: randomly shuffle the input data
        ratio: ratio to be setted beforehand
    Returns:
        sublist_a: the first list after split
        sublist_b: the second list after split
    '''
    raw_data = len(input_data)
    offset = int(raw_data * ratio)
    if raw_data == 0 or offset < 1:
        return [], input_data
    if shuffle:
        random.shuffle(input_data)
    sublist_a = input_data[:offset]
    sublist_b = input_data[offset:]
    return sublist_a, sublist_b

def generate_train_dev():
    # Read the train 5500 label text file
    path = os.path.join(os.getcwd(), "data", "train_5500.label.txt")
    f = open(path)
    lines = f.readlines()
    f.close()

    # split the raw file into train and dev using a ratio of 1:9
    train, dev = random_split(lines, shuffle=True, ratio=0.9)

    # generate the train data
    file = open(config.get('param','path_train'), 'w')
    for i in range(len(train)):
        file.write(train[i])
    file.close()

    # generate the dev data
    file = open(config.get('param','path_dev'), 'w')
    for i in range(len(dev)):
        file.write(dev[i])
    file.close()

if __name__ == '__main__':
    generate_train_dev()