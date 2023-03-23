import pandas as pd

def load_dataset(dataset_number):
    """
    Loads the dataset from the csv file.
    
    Args:
        dataset_number (int): The dataset number (1 or 2).

    Returns:
        pandas.DataFrame: The loaded dataset with labeled columns.
    """
    file_path_1 = '../dataset1.csv'
    file_path_2 = '../dataset2.csv'

    if dataset_number == 1:
        df = pd.read_csv(file_path_1, header=None, names=['x', 'y'])
        
    elif dataset_number == 2:
        df = pd.read_csv(file_path_2, header=None, names=['x', 'y', 'z'])
    else:
        raise ValueError("Invalid dataset number")

    return df


# df = load_dataset(2)
# print(df.head())