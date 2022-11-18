# Python script containing Logistic regression class named 'ishan_J033',
# required as assignment 2 for FOML subject.
# Note: I opted for advanced evaluation. Hence, made a class to perform Logstic regression

import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class ishan_J033:
    """
    Logistic Regression is a very popular algorithm,
    which is often used to predict binary outcomes,
    using previous examples.

    This class can be used to tackle binary classification problems.
    The class is inspired by the Scikit-Learn API format, as it is widely used.
    """

    def __init__(self, learning_rate=0.001, epochs=1000):
        """Method used to initailise object of this class. This method works as a constructor in Python.

        Args:
            learning_rate (float, optional): Learning rate to update weights and bias, to minimise loss using gradient descent. Defaults to 0.001.
            epochs (int, optional): Number of iterations, while training the model. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        """This method is used to train the model on existing examples,
        The method computes weights respective to the features in the data,
        so as to make better predictions, rather than just randomly guessing the outcomes.

        The weights (including the bias term) are updated using infamous gradient descent algorithm.

        Args:
            X (ndarray type): This is the feature matrix of the data used for training purpose. The target values are predicted using this matrix. The datatype can be a pandas.DataFrame or np.ndarray
            y (np.array type): This is the array consisting of target values used for training the algorithm.
        """

        # Initialise weights and theta0s
        m, nx = X.shape
        self.weights_vec = np.zeros(nx)
        self.theta0 = 0

        # Loop to update weights and bias using gradient descent
        for _ in range(self.epochs):
            h_theta = np.dot(X, self.weights_vec)+self.theta0
            y_hat = 1/(1+np.exp(-h_theta))

            dw = (1/m)*np.dot(X.T, (y_hat-y))
            dtheta0 = (1/m)*sum(y_hat-y)

            self.weights_vec = self.weights_vec - self.learning_rate*dw
            self.theta0 = self.theta0-self.learning_rate*dtheta0

    def predict(self, X):
        """This method is used to predict values. This method MUST be run only after running fit method.
        Else the values returned by this method will be redundant. The method uses the weights and bias computed in the
        fit method to predict binary outcomes based on the argument X.

        Args:
            X (pandas.DataFrame or numpy.ndarray): The values which you want the target values of.

        Returns:
            numpy.array: Array of values which computed using the argument X.
        """
        h_theta = np.dot(X, self.weights_vec) + self.theta0
        y_hats = 1/(1+np.exp(-h_theta))

        # Convert probabilities to labels
        y_hat_preds = np.round_(y_hats)
        return y_hat_preds

    @property
    def coef_(self):
        """Returns vector of weights (including bias). This is a property of the class.
        That is, this doesn't have to be used as a function.

        Returns:
            numpy.array: Vector of weights (including bias)
        """
        return np.append(self.weights_vec, self.theta0)

    def evaluate(self, X, y, return_percentage=True):
        """This method is used to evaluate the algorithm. It returns the accuracy score,
        based on predictions and true values. This method can be used to validate the model.
        Learn more about accuracy score: https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

        Args:
            X (pandas.DataFrame or numpy.ndarray): The values of features used to calculate predictions, to later compare the values with y_true
            y (numpy.array): Array of true values of target variable related to the X argument. This array consists of the ground truth labels.

        Returns:
            float: accuracy score of the model based on arguements.
        """
        m = len(y)
        y_preds = self.predict(X)
        if return_percentage:
            return f"{float(np.sum(y == y_preds)/m)*100:.2f}%"
        return float(np.sum(y == y_preds)/m)


def main():
    # Importing libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # Loading a dataset with binary output variable
    ds = load_breast_cancer()
    X, y = ds.data, ds.target

    # Splitting data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Set penalty to 'none' to better compare the predictions
    model = LogisticRegression(penalty='none')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Checking score of Scikit-Learn's model on test set
    print(
        f"Accuracy of Scikit-Learn's Logistic Regression model: {model.score(X_test,y_test)*100:.2f}%")

    print(
        f"R squared of Scikit-Learn's Logistic Regression model:{r2_score(y_test, y_pred)}")

    custom_model = ishan_J033(epochs=10000)
    custom_model.fit(X_train, y_train)

    # Computing accuracy score of custom model
    accuracy = custom_model.evaluate(X_test, y_test, return_percentage=True)
    print(f"Accuracy of custom made model : {accuracy}")


if __name__ == '__main__':
    main()
