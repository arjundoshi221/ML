import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


folder1 = r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\img\Spectacles\spectacles_on"
folder2 = r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\img\Spectacles\spectacles_off"


def load_img(folder):
    # Loading the images and classifying them as per folders
    imgs = []
    y_label = []
    for i in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        imgs.append(img)

        if folder == r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\img\Spectacles\spectacles_on":
            y_label.append(0)
        elif folder == r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\img\Spectacles\spectacles_off":
            y_label.append(1)

    return imgs, y_label


def to_array(folder):
    # Converting the image into an array to prepare for further operations
    arrays, y_label = load_img(folder)

    arrays = np.array(arrays).reshape(-1, 256*256)
    y_label = np.array(y_label).reshape(-1, 1)

    print(arrays.shape, y_label.shape)
    return arrays, y_label


# Making X & Y by combining the folders
x1, y1 = to_array(folder1)
x2, y2 = to_array(folder2)
X = np.concatenate((x1, x2))
Y = np.concatenate((y1, y2))
# print(X.shape,Y.shape)

# Standardize X
sc1 = StandardScaler()
X = sc1.fit_transform(X)

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Model Building
LDA = lda(n_components=1)
LDA.fit(X_train, y_train)
zscore = LDA.transform(X_train)

# Prediction
y_pred = LDA.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
