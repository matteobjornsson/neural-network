from types import new_class
import numpy as np
import math
import TestData
from typing import Callable
import pprint

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
        self.learning_rate = .5
        # weights, biases, and layer outputs are lists with a length corresponding to
        # the number of hidden layers + 1. Therefore weights for layer 0 are found in 
        # weights[0], weights for the output layer are weights[-1], etc. 
        self.weights = self.generate_weight_matrices()
        self.biases = self.generate_bias_matrices()
        # activation_outputs[0] is the input values X, where activation_outputs[1] is the
        # activation values output from layer 1. activation_outputs[-1] represents
        # the final output of the neural network
        self.activation_outputs = [None] * self.layers
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
                weights.append(np.random.randn(layer_nodes, layer_inputs) * 1/layer_inputs)
        self.initial_weights = weights
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
                biases.append(0)
        return biases

    def set_input_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        self.activation_outputs[0] = X
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


    # def tanh(self, z):
    #     """ Return the hyperbolic tangent of z: t(z) = tanh(z)
    #     Input: real number or numpy matrix
    #     Return: real number or numpy matrix.
    #     """
    #     return np.tanh(z)
    
    # def d_tanh(self, z):
    #     """ Return the derivative of tanh: d/dz t(z) = sech^2(z) = 1/cosh^2(z)
    #     Input: real number or numpy matrix
    #     Return: real number or numpy matrix.
    #     """
    #     return 1 / np.square(np.cosh(z))

    ################# COST functions and their derivatives #####################
    # for notation cost function will be noted 'Err()
    def mean_squared_error(self):
        """ takes in matrix(s?), calculates the mean squared error w.r.t. target
        TODO: figure out what inputs are needed and what dimensions. what is return? 
        """
        pass


    ################ FORWARD PASS  ###################################

    def calculate_activation_output(self, W: np.ndarray, X: np.ndarray, b: np.ndarray):
        """ Return A = activation_function(W*X + b)
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        :param activation_function: function for calculating outputs of layer
        """
        Z = np.dot(W, X) + b
        A = self.sigmoid(Z)
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
        for i in range(self.layers):
            if i == 0:
                continue

            print("layer: ", i, " nodes:", self.layer_node_count[i])
            print("previous layer node count:", self.layer_node_count[i-1])
            W = self.weights[i]
            X = self.activation_outputs[i-1]
            b = self.biases[i]
            self.activation_outputs[i] = (
                self.calculate_activation_output(W, X, b)
                )
            print("activation for layer", i, ':\n', self.activation_outputs[i])


        final_estimate = self.activation_outputs[-1]
        print("Forward pass estimate:", final_estimate)
        error = .5 * np.sum(np.square(self.data_labels - final_estimate))
        print("error: ", error)

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

    def calculate_inner_layer_derivative(self, j: int):
        """ Return delta_j = a_j * (1 - a_j) * W^T_j+1 dot delta_j+1
        where j = layer, a_j = activation output matrix of layer j, W_j+1 = weights
        matrix of layer j+1 and delta_j+1 is the derivative matrix of layer j+1.
        Here * means elementwise multiplication, 'dot' means dot product. 

        note: a(1-a) is from the derivative of the sigmoid activation function. 
        This would need to be changed if using a different function. 

        :param j: inner layer for which we are calculating the derivative
        """
        assert j != self.layers-1
        a = self.activation_outputs[j]
        W = self.weights[j+1]
        delta_jPlusOne = self.layer_derivatives[j+1]

        d_layer = (self.d_sigmoid(a) * np.dot(W.T, delta_jPlusOne))

        return d_layer


    def calculate_output_layer_derivative(self, error_fn_name: str, activation_fn_name: str):
        """ Return delta_output = B (a - Y) * a * (1 - a) if error fn == squared error
        and activation fn  == sigmoid. 
        If activation fn == linear, and error == squared error, return delta_output = B (a - Y).
        TODO: figure out how to do cross entropy

        Here a = activation output matrix of output layer, B = number of nodes 
        in output layer (not sure if this is correct), Y = ground truth for input
        examples X. 

        * means elementwise multiplication, 'dot' means dot product. 
        """
        a = self.activation_outputs[-1]
        Y = self.data_labels
        # B = self.layer_node_count[-1]

        if error_fn_name == "squared":
            if activation_fn_name == "linear":
                d_layer = (a - Y)
            elif activation_fn_name == "sigmoid":
                d_layer = (a - Y) * a * (1 - a)
            else:
                raise ValueError("you haven't implemented that yet")
        else:
            raise ValueError("you haven't implemented that yet")

        return d_layer


    def backpropagation_pass(self):
        """ Starting from the input layer propogate the inputs through to the output
        layer.
        """
        for i in reversed(range(1, self.layers)):
            print("backprop layer:", i)
            if i == self.layers - 1:
                self.layer_derivatives[i] = self.calculate_output_layer_derivative("squared", "sigmoid")
            else:
                self.layer_derivatives[i] = self.calculate_inner_layer_derivative(i)

            # update weights
            # W^i_new = W^i_old - dW^i * learning_rate
            # dW^i = delta_i dot a^(i-1).T 
            delta_i = self.layer_derivatives[i]
            a_iMinusOne = self.activation_outputs[i-1].T
            #TODO: add momentum term to update
            print("old weights:\n", self.weights[i])
            dWeight = np.dot(delta_i, a_iMinusOne)
            old_weights = self.weights[i]
            change_weights = dWeight * self.learning_rate
            new_weights = old_weights - change_weights
            self.weights[i] = new_weights
            print('\ngradient for layer ', i,':\n', delta_i)
            print("new weights:\n", self.weights[i])

            # update bias
            # B^i_new = B^i_old - dB^j * learning_rate
            # dB^i = delta_i
            #TODO: add momentum term to update
            self.biases[i] =- delta_i * self.learning_rate

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
    print("input data dimension:", X.shape[0], "# samples:", X.shape[1])
    # X = X[:, 0].reshape(3,1)
    print(X.shape)
    print("labels dimension:", X.shape[0], "# samples:", X.shape[1])
    # X = X[:, 0].reshape(3,1)
    print(labels.shape)
    # X = np.array([[.05],[.10]])
    # labels = np.array([[.01],[.99]])
    input_size = X.shape[0]
    hidden_layers = [input_size]
    regression = False
    output_size = 1
    NN = NeuralNetwork(
        input_size, hidden_layers, regression, output_size
    )
    # NN.weights[1] = np.array([[.15, .20],[.25, .30]])
    # NN.weights[2] = np.array([[.4, .45],[.5, .55]])
    # NN.biases[1] = .35
    # NN.biases[2] = .60
    NN.set_input_data(X, labels)
    # print(vars(NN))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for i in range(10000):
        NN.forward_pass()
        NN.backpropagation_pass()
    weights = NN.initial_weights
    print("X:", X, "Labels: ", labels)
    print("initial weights: ", weights)
