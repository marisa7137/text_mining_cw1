import sys
import pandas as pd
import numpy as np
import csv

def load_data(train_filepath):
    """
      Function:
      load the train_5500.label text into a modified pandas dataframe
      Args:
      train_filepath (str): the file path of messages csv file

      Returns:
      df (DataFrame): A dataframe of contains two columns: categories and question.
      """
    dataframe = []
    csv.register_dialect('skip_space', skipinitialspace=True)
    with open(train_filepath, 'r') as f:
      reader=csv.reader(f , delimiter=' ', dialect='skip_space')
      for item in reader:
          category = item[0]
          question = ' '.join(item[1:])
          dataframe.append(
            { 'category': category,
              'question': question
            }
    )
    df = pd.DataFrame(dataframe)
    return df

if __name__ == '__main__':
    test = load_data('./data/train_5500.label.txt')
    print(test.head(3))
    