### Question classification COMP61332 Text Mining Running Instruction
#### Note: We have already produced the vocablary list using the function inside the text_parser.py also we have produced the modified glove vocab and fine and coarse labels.If you want to test the fucntions, you can call the functions and setting the to file = True

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
### Note (If you use train mode the output file will be the performance and classification results from dev data.If you use test mode, the output file will be the performance and classification results from test data.) (For the dev output file please wait the entie epochs finished to see the final result since it is updating each epoch)
 ```
The macro F1 for this experiment is 0.8574327908216662
 Groud Truth Label    Predict Label        Question            
NUM                  NUM                  How far is it from Denver to Aspen ?
LOC                  LOC                  What county is Modesto , California in ?
HUM                  HUM                  Who was Galileo ?   
DESC                 DESC                 What is an atom ?   
NUM                  NUM                  When did Hawaii become a state ?
NUM                  NUM                  How tall is the Sears Building ?
HUM                  HUM                  George Bush purchased a small interest in which baseball team ?
ENTY                 DESC                 What is Australia 's national flower ?
DESC                 DESC                 Why does the moon turn orange ?
```

3. **Parameter Setting Change**. The user can change the default setting in the corrsponding config file as well as the model's parameters.


