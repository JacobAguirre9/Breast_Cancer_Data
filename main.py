# This is code for Breast Cancer prediction on the commonly used University of Wisconsin-Madison Breast cancer dataset.

# Main Objective-- Develop accurate and precise algorithm to classify breast cancer detection

# <editor-fold desc="Loading Dataset">
print("Hello World")
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# importing our cancer dataset
df = pd.read_csv('wdbc.csv')
# </editor-fold>

# Checking to see the first 5 rows of data are correct
print(df.head(5))


# <editor-fold desc="Statistical Analysis">
# Statistical analysis code block

df.info()
print(df.describe(include="O"))
print(df.diagnosis.value_counts())

# Code for generation of dependent and independent values

unique_diagnosis = df.diagnosis.unique()
print(unique_diagnosis)
# </editor-fold>

# <editor-fold desc="Data Visualization Code">
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(df[cols], hue="diagnosis")
plt.show()
# </editor-fold>

from sklearn.preprocessing import LabelEncoder
# LabelEncoder can be used to normalize labels.
print(df.head(2))
labelencoder_Y = LabelEncoder()
df.diagnosis = labelencoder_Y.fit_transform(df.diagnosis)
print(df.head(2))

print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


print("Goodbye World")
