import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Read CSV
df = pd.read_csv(
    "D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\M2\Datasets_LDA\IRIS\Iris.csv")
df.drop('Id', inplace=True, axis=1)

# EDA
print(df.shape)
print(df.columns)
# Target variable count (n-1) = n_components
print(df['Species'].value_counts())

# Preprocess the dataset and divide into train and test
le = LabelEncoder()
df = df.apply(le.fit_transform)

X = df.drop(['Species'], axis=1)
y = df['Species']

# Standardize X
sc = StandardScaler()
X = sc.fit_transform(X)

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Building
LDA = lda(n_components=2)
LDA.fit(X_train, y_train)
# Z Scores
z = LDA.transform(X_train)

# Predictions
y_pred = LDA.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
