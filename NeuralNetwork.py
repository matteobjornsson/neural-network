from types import new_class
import numpy as np
import math
import TestData
import DataUtility
import pandas as pd
import matplotlib.pyplot as plt

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
        self.learning_rate = .001
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
        self.error_y = []
        self.error_x = []
        self.pass_count = 0

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
                weights.append(np.random.randn(layer_nodes, layer_inputs) * 1/layer_inputs) # or * 0.01
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
        ''' Public method used to set the data input to the network and save the
        ground truth labels for error evaluation. 
        Return: None
        '''
        self.activation_outputs[0] = X
        self.data_labels = labels


    ################# ACTIVATION FUNCTIONS AND DERIVATIVES #####################

    ''' I do not think we will actually use linear activation fn '''
    # def linear(self, z: np.ndarray) -> np.ndarray:
    #     ''' Returns z: s(z) = z
    #     :param z: weighted sum of layer, to be passed through sigmoid fn
    #     Return: z
    #     '''
    #     return z

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        ''' Returns sigmoid function of z: s(z) = (1 + e^(-z))^-1
        :param z: weighted sum of layer, to be passed through sigmoid fn
        Return: matrix 
        '''
        return 1 / (1 + np.exp(-z))


    def d_sigmoid(self, z):
        """ Derivative of the sigmoid function: d/dz s(z) = s(z)(1 - s(z))
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return self.sigmoid(z) * (1-self.sigmoid(z))
    
    
    
    def CrossEntropy(self,Ground_Truth,Estimate): 
        #Calculate the number of rows in the data set 
        Num_Samples = Estimate.shape[1]
        #S
        # output = self.SoftMax(Ground_Truth)
        Logrithmic = Ground_Truth * np.log(Estimate + .000000000000001)
        return  - np.sum(Logrithmic) / Num_Samples
    
    def SoftMax(self,a): 
        soft = np.exp(a - np.max(a))
        return soft/soft.sum()

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

    ################# COST function #####################

    def mean_squared_error(self, ground_truth: np.ndarray, estimate:np.ndarray) -> float:
        """ takes in matrices, calculates the mean squared error w.r.t. target.
        Input matrices must be the same size. 

        :param ground_truth: matrix holding ground truth for each training example
        :param estimate: matrix holding network estimate for each training example
        """
        m = ground_truth.shape[1]
        return (1/m)* np.sum(np.square(ground_truth - estimate))


    ################ FORWARD PASS  ###################################

    def calculate_net_input(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> None:
        """ Return Z = W*X + b
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        Return: None
        """
        Z = np.dot(W, X) + b
        return Z

    def calculate_sigmoid_activation(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> None:
        """ Return A = sigmoid(W*X + b)
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        Return: None
        """
        Z = self.calculate_net_input(W, X, b)
        A = self.sigmoid(Z)
        return A


    def forward_pass(self) -> None:
        """ Starting from the input layer propogate the inputs through to the output
        layer. Return a matrix of outputs.
        Return: None
        """
        # iterate through each layer, starting at inputs
        for i in range(self.layers):
            # the activation output is known for the first layer (input data)
            if i == 0:
                continue

            # weights into layer i
            W = self.weights[i]
            # outputs of previous layer into layer i
            A = self.activation_outputs[i-1]
            # bias of layer i
            b = self.biases[i]
            # Calculate the activation output for the layer, store for later access
            #if this is a classification network and i is the output layer, caclulate softmax
            if self.regression == False and i == self.layers -1:
                self.activation_outputs[i] = (
                    #Calculate the softmax function 
                    self.SoftMax(self.calculate_net_input(W, A, b))
                )
            # otherwise activation is always sigmoid
            else: 
                self.activation_outputs[i] = (
                    self.calculate_sigmoid_activation(W, A, b)
                )
        # output of the network is the activtion output of the last layer
        final_estimate = self.activation_outputs[-1]
        #calculate the error w.r.t. the ground truth
        if self.regression == False: 
            error = self.CrossEntropy(self.data_labels,final_estimate)
        else: 
            error = self.mean_squared_error(self.data_labels, final_estimate)
            
        if self.pass_count > 5:
            self.error_y.append(error)
            self.error_x.append(self.pass_count)
        self.pass_count += 1
        if self.pass_count < 3:
            print("error: ", error)
            print("\n activations:\n")
            for activation in self.activation_outputs:
                print(activation)
            print("\n weights:\n")
            for weight in self.weights:
                print(weight)
        if self.pass_count % 1000 == 0:
            print("error: ", error)
            print("\n activations:\n")
            for activation in self.activation_outputs:
                print(activation)
            print("\n weights:\n")
            for weight in self.weights:
                print(weight)
        # print("Forward pass estimate:", final_estimate, "error: ", error)
        

    ############### BACKPROPAGATION FUNCTION ###################################
    # pseudo code for a single pass of backpropagation: 
    #   calculate the cost function matrix for inputs and weights
    #   for each layer calculate backwards from output layer (n) to input layer (0):
    #       dA_n matrix (dA = dErr/dA) -> derivative of cost function w.r.t. activation outputs from layer n
    #       dW_n matrix (dW = dErr/dW) -> derivative of cost function w.r.t. weights from layer n
    #       db_n matrix (db = dErr/db) -> derivative of cost function w.r.t. bias inputs from layer n
    #   when you have calculated dW_0 and db_0, update weights 
    #       W = W - dW_0 * learning_rate - momentum * dW_0(t-1) (from previous backpropagation iteration)
    #       b = b - db_0 * learning_rate - momentum * db_0(t-1) (from previous backpropagation iteration

    def calculate_inner_layer_derivative(self, j: int) -> None:
        """ Calculates the partial derivative of the error with respect to the current
        layer: delta_j = a_j * (1 - a_j) * W^T_j+1 dot delta_j+1
        where j = layer, a_j = activation output matrix of layer j, W_j+1 = weights
        matrix of layer j+1 and delta_j+1 is the derivative matrix of layer j+1.
        Here * means elementwise multiplication, 'dot' means dot product. 

        note: a(1-a) is from the derivative of the sigmoid activation function. 
        This would need to be changed if using a different function. 

        :param j: inner layer for which we are calculating the derivative
        """
        # check that this is not the output layer (derivative is different there)
        assert j != self.layers-1
        # activation outputs of this layer
        a = self.activation_outputs[j]
        # weights of the next layer
        W = self.weights[j+1]
        # partial derivative of the next layer
        delta_jPlusOne = self.layer_derivatives[j+1]
        # calculate the derivative of this layer and return it
        d_layer = (self.d_sigmoid(a) * np.dot(W.T, delta_jPlusOne))
        return d_layer


    def calculate_output_layer_derivative(self) -> None:
        """ Return delta_output = (a - Y) * a * (1 - a) 
        ***** this assumes squared error fn and sigmoid activation ************
        Here a = activation output matrix of output layer, Y = ground truth for input
        examples X. 
        * means elementwise multiplication, 'dot' means dot product. 
        """
        # activation outputs of the output layer
        a = self.activation_outputs[-1]
        # ground truth for comparing the estimates to
        Y = self.data_labels
        
        # calculate the derivative dError/dactivation * dactivation/dnet
        # here (a - Y) is the error fn derivative, a * (1-a) is the sigmoid derivative
        #TODO: May be the cause if numbers are skewedd 
        if self.regression == False: 
            d_layer = (a - Y)
        else: 
            d_layer = (a - Y) * a * (1 - a)
        return d_layer


    def backpropagation_pass(self):
        """ Starting from the input layer propogate the inputs through to the output
        layer.
        """
        # iterate through each layer, starting from the output layer and calculate
        # the partial derivative of each layer
        for i in reversed(range(1, self.layers)):
            # the last layer is differentiated differently, so it is picked out.
            if i == self.layers - 1:
                self.layer_derivatives[i] = self.calculate_output_layer_derivative()
            else:
                self.layer_derivatives[i] = self.calculate_inner_layer_derivative(i)
            # the goal of backpropagation is to update the weights of each layer.
            # This can be done after calculating the derivative for each layer.
            self.update_weights(i)
            self.update_bias(i)

    def update_bias(self, i: int) -> None:
        """ update the bias. The formula: B^i_new = B^i_old - delta_i * learning_rate
        the partial derivative of the bias is the same as the partial derivtive of
        of the layer.
        :param i: layer for which we are updating the bias
        Return: None
        """
        #grab this layer's derivative
        # print(self.layer_derivatives[i].shape)
        m = self.layer_derivatives[i].shape[1]
        delta_i = (1/m) * np.sum(self.layer_derivatives[i], axis=1, keepdims=True)
        # delta_i = self.layer_derivatives[i]
        #TODO: add momentum term to update
        #update the bias
        self.biases[i] -= delta_i * self.learning_rate
        # print("biases:", self.biases[i])
        # print("--")

    def update_weights(self, i: int) -> None:
        """ update the weights. The formula:  W^i_new = W^i_old - dW^i * learning_rate
        where dW^i = delta_i dot a^(i-1).T 
        the partial derivative of the weights is the partial derivtive of the layer
        times the activation values of the previous layer.
        :param i: layer for which we are updating the bias
        Return: None
        """
        # get the partial derivative of this layer
        delta_i = self.layer_derivatives[i]
        # get the previous layer activation values
        a_iMinusOne = self.activation_outputs[i-1].T
        #TODO: add momentum term to update
        # calculate the change in weights per learning rate
        dWeights = np.dot(delta_i, a_iMinusOne)
        # adjust the weights as the old values minus the derivative times the learning rate
        self.weights[i] -= dWeights * self.learning_rate

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
    X , labels = TD.classification()
    # this code is for testing many points at once from real data
    """
    df = pd.read_csv(f"./TestData/abalone.csv")
    D = df.to_numpy()
    labels = D[:, -1]
    labels = labels.reshape(1, labels.shape[0])
    D = np.delete(D, -1, 1)
    D = D.T
    X = D
    labels = labels.T
    """
    #labels = labels.T
    input_size = X.shape[0]
    hidden_layers = [input_size]
    regression = False
    output_size = 2
    NN = NeuralNetwork(
        input_size, hidden_layers, regression, output_size
    )
    NN.set_input_data(X, labels)
    # print(vars(NN))
    plt.ion()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for i in range(50000):
        NN.forward_pass()
        NN.backpropagation_pass()
        if i % 1000 == 0:
            plt.plot(NN.error_x, NN.error_y)
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
    plt.ioff()
    plt.plot(NN.error_x, NN.error_y)
    plt.show()
