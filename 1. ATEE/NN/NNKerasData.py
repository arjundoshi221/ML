import tensorflow as tf
from keras import models, layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv(
    r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\1. TEE\SVM\diabetes.csv")

# Preprocess the dataset and divide into train and test
le = LabelEncoder()
df = df.apply(le.fit_transform)

# Target & Features
y = df.iloc[:, -1]
X = df.iloc[:, :8]

# Standardize X
sc = StandardScaler()
X = sc.fit_transform(X)

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Binary Model Building

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(512, "relu"),
    layers.Dense(128, activation=layers.LeakyReLU()),
    layers.Dense(1, "sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])


# 3. Fit the model
model.fit(X_train, y_train, epochs=50)

# Predictions
pred = model.predict(X_test)
y_pred = np.where(pred > 0.5, 1, 0)
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
