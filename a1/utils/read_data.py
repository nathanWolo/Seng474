import pandas as pd

'''Reads the data from the csv file and returns a pandas dataframe.'''
def read_data():
    df = pd.read_csv('../cleaned_adult.csv')

    return df



'''Partitions the data into train and test sets, using a taking what 
percentage of the data train / test on as input. Returns the train and test'''
def partition_data(df, train_size=0.75, random_state=99):
    # Split the data into train and test sets. (0.75, 0.25) split.
    train_df = df.sample(frac=train_size, random_state=random_state)
    test_df = df.drop(train_df.index)

    return train_df, test_df

# train, test = partition_data(read_data())
# print(train.shape)
# print(test.shape)