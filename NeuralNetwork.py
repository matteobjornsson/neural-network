import numpy as np
import math
import TestData
from typing import Callable

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
        self.layer_node_count = [input_size] + hidden_layers + [output_size]
        self.layers = len(self.layer_node_count)
        # learning rate
        self.eta = 0.5
        # weights, biases, and layer outputs are lists with a length corresponding to
        # the number of hidden layers + 1. Therefore weights for layer 0 are found in 
        # weights[0], weights for the output layer are weights[-1], etc. 
        self.weights = self.generate_weight_matrices()
        self.biases = self.generate_bias_matrices()
        # layer_outputs[0] is the input values X, where layer_outputs[1] is the
        # activation values output from layer 1. layer_outputs[-1] represents
        # the final output of the neural network
        self.layer_outputs = [None] * self.layers
        self.layer_derivatives = [None] * self.layers
        self.data_labels = None

    ################# INITIALIZATION HELPERS ###################################

    def generate_weight_matrices(self):
        # initialize weights randomly, close to 0
        # generate the matrices that hold the input weights for each layer. Maybe return a list of matrices?
        # will need 1 weight matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        weights = []
        counts = self.layer_node_count
        for i in range(self.layers):
            if i == 0:
                weights.append([])
            else:
                # initialze a (notes, inputs) dimension matrix for each layer. 
                # layer designated by order of append (position in weights list)
                layer_nodes = counts[i]
                layer_inputs = counts[i-1]
                weights.append(np.random.randn(layer_nodes, layer_inputs) * 0.01)
        return weights

    def generate_bias_matrices(self):
        # initialize biases as 0
        # generate the matrices that hold the bias value for each layer. Maybe return a list of matrices?
        # will need 1 bias matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        biases = []
        counts = self.layer_node_count
        for i in range(self.layers):
            if i == 0:
                biases.append([])
            else:
                # initialze a (nodes, 1) dimension matrix for each layer. 
                # layer designated by order of append (position in biases list)
                layer_nodes = counts[i]
                biases.append(np.zeros((layer_nodes, 1)))
        return biases

    def set_input_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        self.layer_outputs[0] = X
        self.data_labels = labels

    ################# ACTIVATION FUNCTIONS AND DERIVATIVES #####################
    def linear(self, z):
        ''' Returns z: s(z) = z
        Input can be a real number or numpy matrix.
        Return: float or matrix 
        '''
        return z

    def sigmoid(self, z):
        ''' Returns sigmoid function of z: s(z) = (1 + e^(-z))^-1
        Input can be a real number or numpy matrix.
        Return: float or matrix 
        '''
        return 1 / (1 + np.exp(-z))

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
        return np.tanh(z)
    
    def d_tanh(self, z):
        """ Return the derivative of tanh: d/dz t(z) = sech^2(z) = 1/cosh^2(z)
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return 1 / np.square(np.cosh(z))

    ################# COST functions and their derivatives #####################
    # for notation cost function will be noted 'Err()
    def mean_squared_error(self):
        """ takes in matrix(s?), calculates the mean squared error w.r.t. target
        TODO: figure out what inputs are needed and what dimensions. what is return? 
        """
        pass


    ################ FORWARD PASS  ###################################
    def calculate_weighted_sum(self, W: np.ndarray, X: np.ndarray, b: np.ndarray):
        """ Return Z = W*X + b
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        """
        print("Shape W: ", W.shape)
        print("Shape X: ", X.shape)
        Z = np.dot(W, X) + b
        return Z

    def calculate_activation_output(self, Z: np.ndarray, activation_function: Callable):
        """ Return A = activation_function(Z)
        :param activation_function: function for calculating outputs of layer
        """
        A = activation_function(Z)
        print("activation function:", activation_function.__name__)
        print(A)
        return A

    def forward_pass(self):
        """ Starting from the input layer propogate the inputs through to the output
        layer. Return a matrix of outputs.
        :param X: training data for the NN
        """
        # this function needs to iterate through each layer and calculate the activation
        # values for each layer. last layer is the output. 
        # probably going to need some if/else blocks to determine how many layers and dimensions of matrices
        # returns the "cost matrix" for all weights and inputs
        for i in range(len(self.layer_outputs)):
            if i == 0:
                continue
            
            Z_i = self.calculate_weighted_sum(
                self.weights[i], 
                self.layer_outputs[i-1], 
                self.biases[i], 
                )

            if i != len(self.layer_outputs)-1:
                print("layer: ", i)
                self.layer_outputs[i] = self.calculate_activation_output(Z_i, self.tanh)            
            
            else:
                if self.regression:
                    activation_fn = self.linear
                else:
                    activation_fn = self.sigmoid

                print("layer: ", i)
                self.layer_outputs[i] = self.calculate_activation_output(Z_i, activation_fn)  

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

if __name__ == '__main__':
    TD = TestData.TestData()
    X , labels = TD.regression()
    print(X.shape)
    X = X.T
    labels = labels.T
    # X = X[:, 0].reshape(3,1)
    print(X.shape)
    input_size = X.shape[0]
    hidden_layers = [input_size + 1]
    regression = True
    output_size = 1
    NN = NeuralNetwork(
        input_size, hidden_layers, regression, output_size
    )
    NN.set_input_data(X, labels)
    # print(vars(NN))
    NN.forward_pass()
