'''
Start code for Project 1-Part 1 and optional 2. 
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: A. Burak Gulhan
    PSU Email ID: abg6029@psu.edu
    Description: (A short description of what each of the functions you're written does).
}
'''


import math
import numpy as np

# TODO: write your MaximalLikelihood class 
class ML():
    def __init__(self, M):
        """
        Initializes the Maximum Likelihood model. You may add any parameters/variables you need.
        You may also add any functions you may need to the class.
        Args:
            _M (int): Degree of polynomial.
            _w (np.array): Predicted weights of model.
            sigma (float): Calculated using beta parameter of the model. Used when graphing.
        """
        self._M = M # dimension of weights
        self._w = None # vector of model weights
        self.sigma = 0.0 # maximum likelihood standard deviation    

    def fit(self, x, y):
        """
        Fits the Maximum Likelihood model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        # calculate weights
        V = np.vander(x, self._M, increasing=True) # store vandermonde matrix for calculating w
        self._w = np.linalg.solve(V.T@V, V.T@y) #
        #self._w = np.linalg.inv(V.T @ V) @ V.T @ y

        # calculate sigma
        N = x.size
        tmp = np.sum(((V@self._w)-y)**2) # calculate Sum from n=1 to N, {y(x_n, w) - y_n}^2 where y() is the polynomial function
        self.sigma = np.asscalar((tmp/N)**0.5) # calculate standard deviation
        print("ML weights")
        print(self._w)

    def predict(self, x):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
        Returns:
            np.array: The predicted targets of the data.
        """
        V = np.vander(x, self._M, increasing=True) # vandermonde matrix
        y = V@self._w
        return y


# TODO: write your MAP class
class MAP():
    def __init__(self, M, alpha=0.005, beta=11.1):
        """
        Initializes the Maximum A Posteriori model. You may add any parameters/variables you need.
        Args:
            alpha (float): The alpha parameter of the model.
            beta (float): The beta parameter of the model.
            _M (int): Degree of polynomial.
            _w (np.array): Predicted weights of model.
            sigma (float): Calculated using beta parameter of the model. Used when graphing.

        You may also add any functions you may need to the class.
        """
        self.alpha = alpha
        self.beta = beta
        self._M = M # dimension of weights
        self._w = None # vector of model weights
        self.sigma = 0.0 # maximum likelihood standard deviation

    def fit(self, x, y):
        """
        Fits the Maximum A Posteriori model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        # calculate weights
        V = np.vander(x, self._M, increasing=True) # store vandermonde matrix for calculating w
        Lambda = self.alpha/self.beta # calculate lambda of error function
        self._w = np.linalg.solve(V.T@V+(np.identity(self._M)*Lambda), V.T@y) # np.linalg.inv((V.T@V)-(np.identity(self._M)*Lambda)) @ V.T @ y
        # calculate sigma
        self.sigma = self.beta**(-0.5)
        print("MAP weights")
        print(self._w)

    def predict(self, x):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
        Returns:
            np.array: The predicted targets of the data.
        """
        V = np.vander(x, self._M, increasing=True) # vandermonde matrix
        y = V@self._w
        return y

# Optional: If you choose to implement a classifier, please do so in this class
class Classifier():
    def __init__(self, params=None):
        """
        Initializes the classifier. You may add any parameters you want. 

        You may also add any functions you may need to the class.
        """
        self.params = params

    def fit(self, x, y):
        """
        Fits the classifier to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Predicts on the data x.
        Args:
            x (np.array): The features of the data.

        Returns:
            out (np.array): target predictions.
        """
        out = x
        raise NotImplemented