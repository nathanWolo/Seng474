import sys
sys.path.append('..')
from utils import read_data
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


train, test = read_data.partition_data(read_data.read_data())

