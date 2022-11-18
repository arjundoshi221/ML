import tensorflow as tf
from keras import models, layers
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
    images, y_label = load_img(folder)

    images = np.array(images).reshape(-1, 256*256)
    y_label = np.array(y_label).reshape(-1, 1)

    print(images.shape, y_label.shape)
    return images, y_label


# Making X & Y by combining the folders
x1, y1 = to_array(folder1)
x2, y2 = to_array(folder2)
X = np.concatenate((x1, x2))
Y = np.concatenate((y1, y2))

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 1. Building the structure
# Multiple
nn_model = models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(5, activation="softmax")
])

# Binary
model = models.Sequential([
    layers.Flatten(),
    layers.Dense(512, "relu"),
    layers.Dense(128, activation=layers.LeakyReLU()),
    layers.Dense(1, "sigmoid")
])
# 2. Compile the model
# Multiple
nn_model.compile(loss="sparse_categorical_crossentropy",
                 optimizer="Adam",
                 metrics=['accuracy'])
# Binary
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",  # or Adam
              metrics=["accuracy"])

# 3. Fit the model
history = nn_model.fit(X_train, y_train, epochs=30)

# Predictions
y_pred = nn_model.predict(X)
cm = confusion_matrix(Y, y_pred)
accuracy_score(Y, y_pred)
