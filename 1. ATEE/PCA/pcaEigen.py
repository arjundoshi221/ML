from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

faces = datasets.fetch_olivetti_faces()
# faces.data.shape

X_train, X_test, y_train, y_test = train_test_split(
    faces.data, faces.target, random_state=0)

# Building PCA Model
pca = PCA(0.95)
pca.fit(X_train)

# Eigen Vectors
eigenvectors = pca.components_

# Variance capture
var = pca.explained_variance_ratio_

# New X_train
omega = pca.fit_transform(X_train)
X_train = omega

# Eigen face
# E = A * V.T
eigenFace = np.dot(omega, eigenvectors)
eigenFace.shape
