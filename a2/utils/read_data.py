import sys
sys.path.append('..')
from fashionmnist.utils import mnist_reader
import numpy as np
import pandas as pd


'''Reads data from fashionmnist and returns a pandas dataframe.'''

def read_data():
    X_train, y_train = mnist_reader.load_mnist('../fashionmnist/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../fashionmnist/data/fashion', kind='t10k')
    
    #rescale the pixels to be between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255

    #normalize feature vectors to have euclidean norm 1
    X_train = X_train / np.linalg.norm(X_train, axis=1).reshape(-1, 1)
    X_test = X_test / np.linalg.norm(X_test, axis=1).reshape(-1, 1)

    train_df = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1))
    test_df = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1))

    print(train_df.shape)
    print(test_df.shape)

    #print(train_df.head())
    return train_df, test_df

train_df, test_df = read_data()

'''For this assignment we're only concerned with classes 0 and 6, so we'll drop the rest of the data.'''

def filter_data(df):
    df = df[df[784].isin([0, 6])]
    print(df.shape)
    for index, row in df.iterrows():
        if row[784] == 6:
            row[784] = 1
    shuffled_df = df.sample(frac=1)
    return shuffled_df

train_df = filter_data(train_df)
test_df = filter_data(test_df)

print(train_df.head())