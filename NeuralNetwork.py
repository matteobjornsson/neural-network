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
        self.weights = self.generate_weight_matrices()
        self.biases = self.generate_bias_matrices()
        # layer_outputs[0] is the input values X, where layer_outputs[1] is the
        # activation values output from layer 1. layer_outputs[-1] represents
        # the final output of the neural network
        self.layer_outputs = []

    ################# INITIALIZATION HELPERS ###################################

    def generate_weight_matrices(self):
        # initialize weights randomly, close to 0
        # generate the matrices that hold the input weights for each layer. Maybe return a list of matrices?
        # will need 1 weight matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        pass

    def generate_bias_matrices(self):
        # initialize biases as 0
        # generate the matrices that hold the bias value for each layer. Maybe return a list of matrices?
        # will need 1 bias matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        pass


    ################# ACTIVATION FUNCTIONS AND DERIVATIVES #####################
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
    # for notation cost function will be noted 'Err()
    def mean_squared_error(self):
        """ takes in matrix(s?), calculates the mean squared error w.r.t. target
        TODO: figure out what inputs are needed and what dimensions. what is return? 
        """
        pass

    ################ FORWARD PASS  ###################################
    def calculate_Activation_Output(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, activation_function: function):
        """ Return A = activation_function(W*X + b)
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        :param activation_function: function for calculating outputs of layer
        """
        return activation_function(W*X + b)

    def forward_pass(self, X: np.ndarray):
        """ Starting from the input layer propogate the inputs through to the output
        layer. Return a matrix of outputs.
        :param X: training data for the NN
        """
        # this function needs to iterate through each layer and calculate the activation
        # values for each layer. last layer is the output. 
        # probably going to need some if/else blocks to determine how many layers and dimensions of matrices
        # returns the "cost matrix" for all weights and inputs
        pass

    ############### BACKPROPAGATION FUNCTION ###################################
    # pseudo code for a single pass of backpropagation: 
    #   calculate the cost function matrix for inputs and weights
    #   for each layer calculate backwards from output layer (n) to input layer (0):
    #       dA_n matrix (dA = dErr/dA) -> derivative of cost function w.r.t. activation outputs from layer n
    #       dW_n matrix (dW = dErr/dW) -> derivative of cost function w.r.t. weights from layer n
    #       db_n matrix (db = dErr/db) -> derivative of cost function w.r.t. bias inputs from layer n
    #   when you have calculated dW_0 and db_0, update weights 
    #       W = W - dW_0 * learning_rate - momentum * (dW_0 from previous backpropagation iteration)
    #       b = b - db_0 * learning_rate - momentum * (db_0 from previous backpropagation iteration

    def backpropagation_pass(self, cost_matrix: np.ndarray):
        """ Starting from the input layer propogate the inputs through to the output
        layer. Return a matrix of outputs.
        :param cost_matrix:
        :param X: 
        """
        pass

    ##################### CLASSIFICATION #######################################
    def classify(self, X: np.ndarray) -> list:
        """ Starting from the input layer propogate the inputs through to the output
        layer. 
        :param X: test data to be classified
        Return: a list of [ground truth, estimate] pairs.
        """
        # basically the same as a forward pass, but return the estimates instead
        # of loss function? 
        pass