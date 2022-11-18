from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd. read_csv(
    r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\1. TEE\1. ATEE\Regression\carPrice.csv")

le = LabelEncoder()
df = df.apply(le.fit_transform)

# Dividing data into X and y variables
X = df.iloc[:, 1:]
y = df.pop('price')

sc = StandardScaler()
X = sc.fit_transform(X)

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))
