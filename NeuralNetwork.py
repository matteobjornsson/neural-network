import numpy as np
import math

class NeuralNetwork:


    def __init__(self, input_size: int, hidden_layers: list,
                    regression: bool, output_size: int) -> None:
        """
        :param input_size: int. dimension of the data set (number of features in x).
        :param hidden_layers: list. [n1, n2, n3..]. List of number of nodes in 
                                each hidden layer. empty list == no hidden layers.
        :param regression: bool. Is this network estimating a regression output?
        :param output_size: int. Number of output nodes (1 for regression, otherwise 1 for each class)
        """ 
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.regression = regression
        self.output_size = output_size
        # learning rate
        self.eta = 0.5

    ################# ACTIVATION functions and their derivatives ###############
    def sigmoid(self, z):
        ''' Returns sigmoid function of z: s(z) = (1 + e^(-z))^-1
        Input can be a real number or numpy matrix.
        Return: float or matrix 
        '''
        return 1 / (1 + math.exp(-z))

    def d_sigmoid(self, z):
        """ Derivative of the sigmoid function: d/dz s(z) = s(z)(1 - s(z))
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def tanh(self, z):
        """ Return the hyperbolic tangent of z: t(z) = tanh(z)
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return math.tanh(z)
    
    def d_tanh(self, z):
        """ Return the derivative of tanh: d/dz t(z) = sech^2(z) = 1/cosh^2(z)
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return 1 / (math.cosh(z)**2)

    ################# COST functions and their derivatives #####################
    def mean_squared_error(self):
        """ takes in matrix(s?), calculates the mean squared error w.r.t. target
        TODO: figure out what inputs are needed and what dimensions. what is return? 
        """
        pass

    ################ FORWARD PASS FUNCTIONS ####################################
    def calculate_Z(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ Return Z = WX+b
        TODO: figure out required dimensions, transpose, etc
        """
        return W * X + b

    def calculate_A(self, Z: np.ndarray, activation_function: function):
        """ Return A = activation_function(Z)
        """
        return activation_function(Z)
    
    ############### BACKPROPAGATION FUNCTION ###################################