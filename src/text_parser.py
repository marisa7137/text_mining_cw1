import numpy as np
from collections import Counter
import torch
import re
import csv
import torch, random
from configparser import ConfigParser
# import the configure files
config = ConfigParser()
config.read("src/bow.config")

# Added the random seed generator
torch.manual_seed(6666666) 
random.seed(666666)

class TextParser:
    def __init__(self):
        self.raw_text_path = config.get("param","train_5500")  # Path for the raw text
        self.stopwords_path = config.get("param","stop_words")   # Path for the stopwords collection text
        self.training_set_path = ''  # Path for the split training set
        self.dev_set_path = ''  # Path for the developing set
        self.fine_labels_path = ''  # Path for the label collection
        self.vocab_path = ''  # Path for the vocabulary collection
        self.stopwords = []  # list of a collection of stopwords
        self.raw_sentences = []
        self.fine_pair = []  # a raw pair is in the format of (label, sentence) where both are strings
        self.coarse_pair = []
        self.fine_labels = []
        self.coarse_labels = []
        self.sentences = []
        self.indexed_sentence_pair = []
        self.tokens = [] # a list of tokens
        self.embedded_data = []
        self.vocab_count_dict = None
        self.vocab = []  # a vocabulary list contains all words in the document
        self.words = []
        self.split_radio = 0.1  # for splitting the data into developing set and training set
        self.indexed_sentence_pair = []

        self.load_stopwords()
        self.load_raw_text()
        self.create_vocab(to_file=False)
        self.create_label()


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
        fine_pair: A pair of label and clean tokens format from the raw text_file
        coarse_pair: A pair of coarse label and clean tokens format from the raw text_file
        """

        csv.register_dialect('skip_space', skipinitialspace=True)
        with open(self.raw_text_path, 'r') as f:
             reader= csv.reader(f , delimiter=' ', dialect='skip_space')
             for item in reader:
                 label = item[0]
                 fine_class_label = label.split(":")[0]
                 question = ' '.join(item[1:])
                 self.raw_sentences.append(question)
                 sentence = self.remove_stopwords(question)
                 sentence = re.sub(r"[^a-zA-Z0-9]", ' ', sentence)
                 tokens = sentence.strip().split(' ')
                 clean_tokens = [token for token in tokens if token != ""]
                 for word in clean_tokens:
                     self.words.append(word)
                 self.fine_pair.append((label, clean_tokens))
                 self.coarse_pair.append((fine_class_label,clean_tokens))
              
                 
    def remove_stopwords(self, sentence):
        """
        Function:
        Remove the stopwords from the given sentence
        Args:
        self (text parser itself), sentence (the given sentence)
        Returns:
        result: a list contain the sentence after removing the stop words.
        
        """
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
                self.fine_labels.append(line)

    def create_vocab(self, to_file=True):
        if len(self.words) > 0:
            self.vocab_count_dict = Counter(self.words)
            self.vocab = sorted(self.vocab_count_dict, key=self.vocab_count_dict.get, reverse=True)
            self.vocab.append('#unk#')
            if to_file:
                with open(self.vocab_path, 'w') as f:
                    for word in self.vocab:
                        f.write(word + '\n')

    def create_label(self):
        if len(self.fine_pair) > 0:
            label_list = [lb[0] for lb in self.fine_pair]
            fine_labels = Counter(label_list)
            self.fine_labels = sorted(fine_labels, key=fine_labels.get, reverse=True)
        if len(self.coarse_pair) > 0:
            coarse_labels_list = [lb[0] for lb in self.coarse_pair]
            coarse_labels = Counter(coarse_labels_list)
            self.coarse_labels = sorted(coarse_labels , key=coarse_labels .get, reverse=True)
            
            

    def create_split_data(self):
        random_indices = np.random.permutation(len(self.raw_sentences))
        dev_size = int(len(self.raw_sentences) * self.split_radio)
        dev_indices = random_indices[:dev_size]
        train_indices = random_indices[dev_size:]
        train_data = [self.raw_sentences[i] for i in train_indices]
        dev_data = [self.raw_sentences[i] for i in dev_indices]
        with open(self.training_set_path, 'w') as f1:
            for i in range(len(train_data)):
                f1.write(train_data[i].strip('\n') + '\n')
        with open(self.dev_set_path, 'w') as f2:
            for j in range(len(dev_data)):
                f2.write(dev_data[j].strip('\n') + '\n')
                
    def get_word_indices(self,type,dim):
            for pair in self.fine_pair:
                if type == "coarse":
                    label = pair[0].split(":")[0]
                    label_embedded = np.int32(self.coarse_labels.index(label))
                # fine pair word
                else:
                    label = pair[0]
                    label_embedded = np.int32(self.fine_labels.index(label))
                    
                sentence = pair[1]
                word_vec = np.zeros(dim)
                for i in range(dim):
                    if i < len(sentence):
                        word = sentence[i]
                        if sentence[i] in self.vocab:
                            word_vec[i] = np.int32(self.vocab.index(word))
                        else:
                            word_vec[i] = np.int32(self.vocab.index('#unk#'))
                self.indexed_sentence_pair.append((label_embedded, torch.LongTensor(word_vec)))
            return self.indexed_sentence_pair






