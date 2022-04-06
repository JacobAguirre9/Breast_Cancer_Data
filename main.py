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

print(len(confusion_matrixs))
plt.figure(figsize=(10, 2))
# plt.title("Confusion Metric Graph")


for index, cm in enumerate(confusion_matrixs):
       plt.xlabel("Negative Positive")
       plt.ylabel("True Positive")

       # Show The Metrics Graph
       cm_metrix_graph(cm)  # Call the Confusion Metrics Graph
       plt.tight_layout(pad=True)

print(df_pred)

# Now let's see which model performed best s.t. we see highest accuracy
df_pred.sort_values('score', ascending=False)
df_pred.sort_values('accuracy_score', ascending=False)

print(df_pred)

# </editor-fold>

# <editor-fold desc="K-fold application">
print(len(df))
# Sample For testing only

cv_score = cross_validate(LogisticRegression(), X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)

pd.dataframe(cv_score).describe().T


def cross_val_scorring(model):
    #     (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )

    model.fit(df[prediction_feature], df[targeted_feature])

    # score = model.score(X_train, y_train)

    predictions = model.predict(df[prediction_feature])
    accuracy = accuracy_score(predictions, df[targeted_feature])
    print("\nFull-Data Accuracy:", round(accuracy, 2))
    print("Cross Validation Score of'" + str(name), "'\n")

    # Initialize K folds.
    kFold = KFold(n_splits=5)  # define 5 different data folds

    err = []

    for train_index, test_index in kFold.split(df):
        # print("TRAIN:", train_index, "TEST:", test_index)

        # Data Spliting via fold indexes
        X_train = df[prediction_feature].iloc[train_index,
                  :]  # train_index = rows and all columns for Prediction_features
        y_train = df[targeted_feature].iloc[train_index]  # all targeted features trains

        X_test = df[prediction_feature].iloc[test_index, :]  # testing all rows and cols
        y_test = df[targeted_feature].iloc[test_index]  # all targeted tests

        # Again Model Fitting
        model.fit(X_train, y_train)

        err.append(model.score(X_train, y_train))

        print("Score:", round(np.mean(err), 2))


        for name, model in zip(list(models_list.keys()), list(models_list.values())):
            cross_val_scorring(model)
# </editor-fold>


# <editor-fold desc="Hypertuning data">
from  sklearn.model_selection import GridSearchCV

# Let's Implement Grid Search Algorithm

# Pick the model
model = DecisionTreeClassifier()

# Tunning Params
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10],
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10) # For 10 Cross-Validation

gsc.fit(X_train, y_train) # Model Fitting

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parameters are")
print(gsc.best_params_)

# Pick the model
model = KNeighborsClassifier()


# Tuning Parameters
param_grid = {
    'n_neighbors': list(range(1, 30)),
    'leaf_size': list(range(1,30)),
    'weights': [ 'distance', 'uniform' ]
}


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10)

# Model Fitting
gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)

# Pick the model
model = SVC()


# Tuning Parameters
param_grid = [
              {'C': [1, 10, 100, 1000],
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000],
               'gamma': [0.001, 0.0001],
               'kernel': ['rbf']
              }
]


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parameters are")
print(gsc.best_params_)

# Pick the model
model = RandomForestClassifier()


# Tuning Parameters
random_grid = {'bootstrap': [True, False],
 'max_depth': [40, 50, None], # 10, 20, 30, 60, 70, 100,
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2], # , 4
 'min_samples_split': [2, 5], # , 10
 'n_estimators': [200, 400]} # , 600, 800, 1000, 1200, 1400, 1600, 1800, 2000

# Implement GridSearchCV
gsc = GridSearchCV(model, random_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parameters are")
print(gsc.best_params_)

model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

import pickle as pkl
# Trained Model # You can also use your own trained model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

filename = 'logistic_model.pkl'
pkl.dump(logistic_model, open(filename, 'wb')) # wb means write as binary

# To read model from file
# load the model from disk
loaded_model = pkl.load(open(filename, 'rb')) # rb means read as binary
result = loaded_model.score(X_test, Y_test)
# </editor-fold>



print("Goodbye World")
