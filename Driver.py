
import numpy as np
import pandas as pd
import TestData
import NeuralNetwork
import matplotlib.pyplot as plt


# set arguments such as how many hidden layers, how many nodes per hidden layer
# identify if the data set is regression, if not, how many classes? 
# init a neural network with these input parameters
# chunk training data into batches (?)
# repeat until error value converges (?)
#   pass forward and backprop on each batch of input X
# once Neural network converges, stop training
# pass in test data on the network and get classification results
# run the results processing on data (mean squared error, F1, etc)
data_sets = ["abalone","Cancer","glass","forestfires","soybean","machine"] 

regression_data_set = {
    "soybean": False,
    "Cancer": False,
    "glass": False,
    "forestfires": True,
    "machine": True,
    "abalone": True
}

# TD = TestData.TestData()
# X , labels = TD.regression()
for data_set in data_sets:
    if data_set != 'machine':
        continue
    regression = regression_data_set[data_set]
    # if regression == False:
    #     continue
    df = pd.read_csv(f"./TestData/{data_set}.csv")
    D = df.to_numpy()
    labels = D[:, -1]
    labels = labels.reshape(1, labels.shape[0])
    D = np.delete(D, -1, 1)
    X = D.T

    # print("input data: ", X.shape, '\n', X)
    # print("input labels: ", labels.shape, '\n', labels)

    input_size = X.shape[0]
    hidden_layers = [input_size-1]
    

    if regression == True:
        output_size = 1
    else:
        output_size = df.Class.nunique()

    NN = NeuralNetwork.NeuralNetwork(
        input_size, hidden_layers, regression, output_size
    )
    NN.set_input_data(X, labels)
    # print(vars(NN))
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ { data_set } $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    plt.ion()
    epochs = 100
    for i in range(epochs):
        NN.forward_pass()
        NN.backpropagation_pass()
        if i % int(epochs/10) == 0:
            plt.plot(NN.error_x, NN.error_y)
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
    plt.ioff()
    plt.plot(NN.error_x, NN.error_y)
    plt.show()
    print("\n Labels: \n",labels)