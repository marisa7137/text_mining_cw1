import numpy as np
from collections import Counter
import torch
import re

"""
text_parser class 
    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
"""



class TextParser:
    def __init__(self, config=None):
        self.raw_text_path = './data/train_5500.label.txt'  # Path for the raw text
        self.stopwords_path = './data/stopwords.txt'  # Path for the stopwords collection text
        self.training_set_path = ''  # Path for the split training set
        self.dev_set_path = ''  # Path for the developing set
        self.labels_path = ''  # Path for the label collection
        self.vocab_path = ''  # Path for the vocabulary collection
        self.stopwords = []  # list of a collection of stopwords
        self.raw_sentences = []
        self.raw_pair = []  # a raw pair is in the format of (label, sentence) where both are strings
        self.labels = []
        self.sentences = []
        self.embedded_data = []
        self.vocab = []  # a vocabulary list contains all words in the document
        self.words = []
        self.split_radio = 0.1  # for splitting the data into developing set and training set

        self.load_stopwords()
        self.load_raw_text()
        self.create_vocab(to_file=False)
        self.create_label(to_file=False)


    def load_stopwords(self):
        """
        Function: 
        load the words from stopwords path and append the words into the stop words list
        Args: 
        self (text parser itself)
        Returns:
        self.stopwords: a list of stopwords coming from the stopwords path file
        """
        with open(self.stopwords_path, 'r') as f:
            for line in f:
                line = line.lower().strip('\n')
                self.stopwords.append(line)

    def load_raw_text(self):
        """
        Function:
        load the train_5500.label file and process it into a raw pair format after lowering and removing the unneccary
        punctions
        Args:
        self (text parser itself)
        Returns:
        raw_pair: A pair of label and sentence format from the raw text_file   
        """
        with open(self.raw_text_path, 'r') as f:
            for line in f:
                self.raw_sentences.append(line.strip('\n') + '\n')
                line = line.lower().strip('\n')
                pair = line.split(' ', 1)
                label = pair[0]
                sentence = self.remove_stopwords(pair[1])
                sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
                for word in sentence.split(' '):
                    self.words.append(word)
                self.raw_pair.append((label, sentence))
                
        

    def remove_stopwords(self, sentence):
        words = sentence.split()
        result_words = [word for word in words if word.lower() not in self.stopwords]
        result = ' '.join(result_words)
        return result

    def load_vocab_from_file(self):
        with open(self.vocab_path, 'r') as f:
            for line in f:
                line = line.lower().strip('\n')
                self.vocab.append(line)

    def load_label_from_file(self):
        with open(self.vocab_path, 'r') as f:
            for line in f:
                line = line.lower().strip('\n')
                self.labels.append(line)

    def create_vocab(self, to_file=True):
        if len(self.words) > 0:
            vocab = Counter(self.words)
            self.vocab = sorted(vocab, key=vocab.get, reverse=True)
            self.vocab.append('')
            self.vocab.append('#unk#')
            if to_file:
                with open(self.vocab_path, 'w') as f:
                    for word in self.vocab:
                        f.write(word + '\n')

    def create_label(self, to_file=True):
        if len(self.raw_pair) > 0:
            label_list = [lb[0] for lb in self.raw_pair]
            labels = Counter(label_list)
            self.labels = sorted(labels, key=labels.get, reverse=True)
            if to_file:
                with open(self.labels_path, 'w') as f:
                    for label in self.labels:
                        f.write(label + '\n')

    def create_split_data(self):
        random_indices = np.random.permutation(len(self.raw_sentences))
        dev_size = int(len(self.raw_sentences) * self.split_radio)
        dev_indices = random_indices[:dev_size]
        train_indices = random_indices[dev_size:]
        train_data = self.raw_sentences[train_indices]
        dev_data = self.raw_sentences[dev_indices]
        with open(self.training_set_path, 'w') as f1:
            for i in range(len(train_data)):
                f1.write(train_data[i].strip('\n') + '\n')
        with open(self.dev_set_path, 'w') as f2:
            for j in range(len(dev_data)):
                f2.write(dev_data[j].strip('\n') + '\n')

    def random_initialise_embedding(self, dim):
        for pair in self.raw_pair:
            label = pair[0]
            sentence = pair[1].lower().split(' ')
            word_vec = np.zeros(dim)
            label_embedded = np.int32(self.labels.index(label))
            for i in range(dim):
                if i < len(sentence):
                    word = sentence[i]
                    if word in self.vocab:
                        word_vec[i] = np.int32(self.vocab.index(word) + 1)
                    else:
                        word_vec[i] = np.int32(self.vocab.index('#unk#') + 1)
            self.embedded_data.append((label_embedded, torch.LongTensor(word_vec)))
        return self.embedded_data
