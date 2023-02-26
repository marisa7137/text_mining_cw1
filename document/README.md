### Question classification COMP61332 Text Mining Running Instruction

The whole document contains three sub folders:
1. **data**: The data folder contains the config files, datasets (train.txt, dev.txt, test.txt, train_550.label.txt), output files, preprocessing files ( vocab, stopwords and labels .etc.)
2. **document**: The document folder contains the description of the function and the how to run
3. **src**: The src folder contains the main code of the project

#### Steps:
1. Locate the current running path under the 'src' folder through terminal
```
    cd text_mining_cw1/src
```
2. The program supports the user in selecting a different combination of training models and word embeddings. The user must input the four augments [train/test,bow/bilistm,fine/coarse,pretrain] needed to run the proposed program.

```
    python question_classifier.py --[train|test] --config [ConfigPath] 
```
| Model | Word Embedding | Class Labels |
| -------- | -------- | -------- |
| Bag of Words | Randomly initialised word embeddings | Fine Class |
| Bidirectional LSTM | Pre-trained word embeddings | Coarse Class |

```
# Train a bilstm model using the hyperparameters in bilstm configuration file for "fine" labels using pretrain word embeddings
    python question_classifier.py --train --config "../data/bilstm.config" 

# Test a bilstm model using the saved model from training process and save the results to the "fine_output.txt"
    python question_classifier.py --test --config "../data/bilstm.config" 

# Train a bag of words model using the hyperparameters in bow congiguration file for "coarse" labelsing without using pretrain word embeddings
    python3 question_classifier.py --train --config '../data/bow.config' 

# Test a bow model using the saved model from training process and save the results to the "coarse_output.txt"
    python question_classifier.py --test --config "../data/bow.config" 
```
After running the code above: the corresponding setting will display on the screen:

```
Pretrain Option: On
Mode: Training
Class Label: fine
Model: bilstm
```
```
Pretrain Option: On
Mode: Testing
Class Label: fine
```
And the result will store as below:
```
Groud Truth Label    Predict Label        Question            
NUM:dist             NUM:dist             How far is it from Denver to Aspen ?
LOC:city             LOC:other            What county is Modesto , California in ?
HUM:desc             HUM:desc             Who was Galileo ?   
DESC:def             DESC:def             What is an atom ?   
NUM:date             NUM:date             When did Hawaii become a state ?
NUM:dist             NUM:dist             How tall is the Sears Building ?
```

3. **Parameter Setting Change**. The user can change the default setting in the corrsponding config file as well as the model's parameters.


