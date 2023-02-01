import pandas as pd


def read_data():
    df = pd.read_csv('./cleaned_adult.csv')

    return df

