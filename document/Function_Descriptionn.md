### train_text_split.py

##### This python file mainly create train and dev output files using the train_5500.label

```python
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
def generate_train_dev():
    """
    Generate the train, dev files and save to the corrosponding path
    """
```

### text_parser.py

##### This python file mainly loads the corresponding dataset, do the preprocessing steps (lower, removing stop words,tokens), create vocabulary list and create different labels files and save files, create pretrain vocab and weights as well as word vectors.

```python
def load_stopwords(self):
    """
    Function:
    load the words from stopwords path and append the words into the stop words list
    Args:
    self (text parser itself)
    Returns:
    self.stopwords: a list of stopwords coming from the stopwords path file
    """
def load_raw_text(self):
    """
    Function:
        load the data file and process it into a raw pair format after tokenlise and lowering and removing the unneccary
    punctions
    Args:
        self (text parser itself)
    Returns:
        fine_pair: A pair of label and clean tokens format from the raw text_file
        coarse_pair: A pair of coarse label and clean tokens format from the raw text_file
    """
def remove_stopwords(self, sentence):
    """
    Function:
        Remove the stopwords from the given sentence
    Args:
        self (text parser itself), sentence (the given sentence)
    Returns:
        result: a list contain the sentence after removing the stop words.
    """
def load_vocab_from_file(self):
    """
    Function:
        load the vocab from the given file
    Args:
        self (text parser itself)
    """
def load_fine_label_from_file(self):
    """
    Function:
        load the fine labels from the given file
    Args:
        self (text parser itself)
    """

def load_coarse_label_from_file(self):
    """
    Function:
        load the coarse labels from the given file
    Args:
        self (text parser itself)
    """
def create_vocab(self, to_file):
    """
    Function:
        Create Vocabulary List and adds padding at the start of the file and #unk# at the end of the file
    Args:
        self (text parser itself)
        to_file(bool): wether save the vocab to the file
    """
def create_label(self, to_file):
    """
    Function:
        Create label Lists(fine and coarse) and save to the files.
    Args:
        self (text parser itself)
        to_file(bool): wether save the vocab to the file
    """
def get_word_indices(self, type, dim, from_file):
    """
    Function:
        Create word index and save as [label, tokens] with associate index number from the given vocab and label file
    Args:
        self (text parser itself)
        type(str): fine or coarse
        dim(int): the demonsion of the word vectoer
        from_file(bool): whether parse the data from previous generate files.
    """
def get_word_indices_from_glove(self, type, dim):
    """
    Function:
        Create word index and save as [label, tokens] with associate index number from the pretrain glove file
    Args:
        self (text parser itself)
        type(str): fine or coarse
        dim(int): the demonsion of the word vectoer
    """
def create_vocab_and_weight_from_pretrained_glove(self):
    """
    Function:
        Create vocabulary list and weights from pretrain glove and save to the corrsponding file
    Args:
        self (text parser itself)
    """
```

### word_embedding.py

#### This python file mainly create the word embedding using either random or pretrain

```python
class Word_Embedding(nn.Module):
    """
    Class:
        Word_Embedding: the class to create Randomly initialised word embeddings/ Pre-trained word embeddings
    forward:
        forward the embedding with word_indices
    """
```

### sentence_rep.py

#### This python file mainly uses the bag of words method to represent the sentences.

```python
class Sentence_Rep(nn.Module):
     """
    Class: Sentence_Rep Class to initial the word representation of bag of words
    """
def forward(self, word_vecs):
    """
    Function:
        addes up each word vector in the sentence list and take the mean (average of the words in a sentence)
    Args:
        self(text parser),
        word_vecs: word vectors
    Return:
        return a out with [batch dim, word embediing]
    """
```

### bilstm.py

#### This python file mainly creates bilstm model

```python
class Model(nn.Module):
    '''
    This class that builds the BiLSTM model with a classifier.
    :param list pre_train_weight: the pre-trained weights
    :param int vocab_size: the size of vocabulary in text parser
    :param int embedding_dim: the embedding dimension (suggested 300 at least)
    :param bool from_pre_train: True if use the pre-trained wrights
    :param bool freeze: True if freeze the weights
    :param bool bow: False if builds BiLSTM
    :param int hidden_dim_bilstm: the hidden dimension of BiLSTM
    :param int hidden_layer_size: the hidden layer size of BiLSTM
    :param int num_of_classes: the number of classes, 6 or 50
    :return: a BiLSTM model with a classifier
    '''
```

### bilstm_train.py

#### This python file mainly define the requiremnts of train of bilstm

```python
def development(batch_size, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s):
    '''
    The function for model development (fine-tuning/optimisation) at the end of each epoch
    :param int batch_size: the developing batch size
    :param DataLoader dev_loader: the developing data loader
    :param Model model: the model updated in the current epoch
    :param torch.nn.NLLLoss loss_function: the loss function
    :param list dev_losses: the developing loss list
    :param list dev_accs: the developing accuracy list
    :param list dev_F1s: the developing F1 score list
    :return: dev_losses, dev_accs, dev_F1s
    '''
def train(t, train_data, dev_data, num_classes, pretrain, lr, epoch, batch, embedding_dim, hidden_dim, hidden_layer,freeze,pre_trained_weight=None):
    '''
    The main function for training
    :param TextParser t: test parser
    :param list train_data: the training data
    :param list dev_data: the developing data
    :param int num_classes: the number of classes, 6 or 50
    :return: None
    '''
```
### bilstm_test.py
#### This python file mainly define the requiremnts of test of bilstm
```python
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
def test(t, test_data, num_classes, model_pth, output_pth):
    '''
    The main function for testing
    :param list test_data: the test data
    :param int num_classes: the number of classes, 6 or 50
    :param str model_pth: the path of trained model file
    :return: None
    '''
```
### bow.py
#### This python file mainly define the requiremnts of model of bag of words

```python
class Model(torch.nn.Module):
    '''
    This class that builds the BOW model with a classifier.
    :param list pre_train_weight: the pre-trained weights
    :param int vocab_size: the size of vocabulary in text parser
    :param int embedding_dim: the embedding dimension (suggested 300 at least)
    :param bool from_pre_train: True if use the pre-trained wrights
    :param bool freeze: True if freeze the weights
    :param bool bow: False if builds BOW
    :param int hidden_dim_bilstm: the hidden dimension of BOW
    :param int hidden_layer_size: the hidden layer size of BOW
    :param int num_of_classes: the number of classes, 6 or 50
    :return: a BOW model with a classifier
    '''
```
### bow_train.py
#### This python file mainly defines the requirement of training part of bag of words
```python
def development(batch_size, dev_loader, model, loss_function, dev_losses, dev_accs, dev_F1s):
    '''
    The function for model development (fine-tuning/optimisation) at the end of each epoch
    :param int batch_size: the developing batch size
    :param DataLoader dev_loader: the developing data loader
    :param Model model: the model updated in the current epoch
    :param torch.nn.NLLLoss loss_function: the loss function
    :param list dev_losses: the developing loss list
    :param list dev_accs: the developing accuracy list
    :param list dev_F1s: the developing F1 score list
    :return: dev_losses, dev_accs, dev_F1s
    '''

def train(t, train_data, dev_data, num_classes, pretrain, lr, epoch, batch, embedding_dim, freeze, hidden_dim, hidden_layer, pre_trained_weight=None):
    '''
    The main function for training
    :param TextParser t: test parser
    :param list train_data: the training data
    :param list dev_data: the developing data
    :param int num_classes: the number of classes, 6 or 50
    :return: None
    '''
```
### bow_test.py
#### This python file mainly defines the requiremnts of the test part of the bag of words
```python
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
def test(t,test_data, num_classes, model_pth,output_pth):
    '''
    The main function for testing
    :param list test_data: the test data
    :param int num_classes: the number of classes, 6 or 50
    :param str model_pth: the path of trained model file
    :return: None
    '''

```
### question_classifier.py
#### This is the main python file to run the entire question classifier system.
```python
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

```




