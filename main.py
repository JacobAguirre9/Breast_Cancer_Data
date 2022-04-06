# This is code for Breast Cancer prediction on the commonly used University of Wisconsin-Madison Breast cancer dataset.

# Main Objective-- Develop accurate and precise algorithm to classify breast cancer detection

print("Hello World")
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# importing our cancer dataset
dataset = pd.read_csv(breast-cancer-wisconsin.csv)
X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values

dataset.head()


print("Goodbye World")
