from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

df = pd.read_csv(
    r"D:\Arjun 2\Coding Files-Arjun\Computer Science\College\IntroToML\1. TEE\PCA\PCA_case study.csv")

df = df.iloc[:, 5:-1].dropna()

# Building PCA Model
pca = PCA(0.95)
pca.fit(df)

# Eigen Vectors
eigenvectors = pca.components_

# Variance capture
var = pca.explained_variance_
varRatio = pca.explained_variance_ratio_

X = pca.fit_transform(df)

print(eigenvectors.T)
print(varRatio)
# print(X)
