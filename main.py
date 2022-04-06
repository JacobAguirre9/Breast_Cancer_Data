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

# <editor-fold desc="Label Creation">
from sklearn.preprocessing import LabelEncoder
# LabelEncoder can be used to normalize labels.
print(df.head(2))
labelencoder_Y = LabelEncoder()
df.diagnosis = labelencoder_Y.fit_transform(df.diagnosis)
print(df.head(2))

print(df.diagnosis.value_counts())
print("\n", df.diagnosis.value_counts().sum())
# </editor-fold>

# <editor-fold desc="Correlation Matrix">
cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concativity_mean',
       'concavepoints_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
corrmatrix = df[cols].corr()
print(corrmatrix)
plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(df[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);
plt.figure(figsize=(15, 10))


fig = px.imshow(df[cols].corr());
fig.show()
# </editor-fold>

# <editor-fold desc="Model Implementation Code">
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics

prediction_feature = ["radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concavepoints_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)
X = df[prediction_feature]
X
print(X.shape)
print(X.values)
y = df.diagnosis
y
print(y.values)

# Now, split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)
print(X_train)
print(X_test)

# Scale the data to keep all the values in the same magnitude of 0 -1

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# </editor-fold>

# <editor-fold desc="ML model selection and model prediction">
def model_building(model, X_train, X_test, y_train, y_test):
       """

       Model Fitting, Prediction And Other stuff
       return ('score', 'accuracy_score', 'predictions' )
       """

       model.fit(X_train, y_train)
       score = model.score(X_train, y_train)
       predictions = model.predict(X_test)
       accuracy = accuracy_score(predictions, y_test)

       return (score, accuracy, predictions)

models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}

print(models_list)
print(list(models_list.keys()))
print(list(models_list.values()))

print(zip(list(models_list.keys()), list(models_list.values())))


def cm_metrix_graph(cm):
       sns.heatmap(cm, annot=True, fmt="d")
       plt.show()


df_prediction = []
confusion_matrixs = []
df_prediction_cols = ['model_name', 'score', 'accuracy_score', "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):
       (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test)

       print("\n\nClassification Report of '" + str(name), "'\n")

       print(classification_report(y_test, predictions))

       df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])

       # For Showing Metrics
       confusion_matrixs.append(confusion_matrix(y_test, predictions))

df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)

# </editor-fold>

print("Goodbye World")
