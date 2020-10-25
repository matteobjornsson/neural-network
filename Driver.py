
import numpy as np
import pandas as pd
import math, random, copy
import TestData
import DataUtility
import NeuralNetwork
import matplotlib.pyplot as plt
import time 
import Performance

# set arguments such as how many hidden layers, how many nodes per hidden layer
# identify if the data set is regression, if not, how many classes? 
# init a neural network with these input parameters
# batch training data into batches (?)
# repeat until error value converges (?)
#   pass forward and backprop on each batch of input X
# once Neural network converges, stop training
# pass in test data on the network and get classification results
# run the results processing on data (mean squared error, F1, etc)

# this function batches data for training the NN, batch size is thie important input parmameter
def batch_input_data(X: np.ndarray, labels: np.ndarray, batch_size: int) -> list:
    batches = []
    # grabs indices of all data points to train on
    data_point_indices = list(range(X.shape[1]))
    # shuffles them
    random.shuffle(data_point_indices)
    # print(data_point_indices)
    # then batches them in a list of [batch, batch labels] pairs
    for i in range(math.ceil(X.shape[1]/batch_size)):
        if i == math.ceil(X.shape[1]/batch_size) - 1:
            batch_indices = data_point_indices
        else:
            batch_indices = data_point_indices[:batch_size]
            data_point_indices = data_point_indices[batch_size:]
        # print(batch_indices)
        # batch indices is an array, selecting all columns of indices in that array
        X_i = X[:, batch_indices]
        labels_i = labels[:, batch_indices]
        batches.append([X_i, labels_i])
    return batches

data_sets = ["abalone","Cancer","glass","forestfires","soybean","machine"] 

regression_data_set = {
    "soybean": False,
    "Cancer": False,
    "glass": False,
    "forestfires": True,
    "machine": True,
    "abalone": True
}
categorical_attribute_indices = {
    "soybean": [],
    "Cancer": [],
    "glass": [],
    "forestfires": [],
    "machine": [],
    "abalone": []
}

for data_set in data_sets:
    if data_set != 'machine':
        continue

    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    # ten fold data and labels is a list of [data, labels] pairs, where 
    # data and labels are numpy arrays:
    tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

    # execute driver for each of the ten folds
    for j in range(10):
        test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[j])
        #Append all data folds to the training data set
        remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
        remaining_labels = [x[1] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
        #Store off a set of the remaining dataset 
        X = np.concatenate(remaining_data, axis=1) 
        #Store the remaining data set labels 
        labels = np.concatenate(remaining_labels, axis=1)
        #Test data print to the scren 
        print("data:", X.shape, '\n', X)
        print()
        print("labels:", labels.shape, '\n', labels)
        regression = regression_data_set[data_set]
        #If the data set is a regression dataset
        if regression == True:
            #The number of output nodes is 1 
            output_size = 1
        #else it is a classification data set 
        else:
            #Count the number of classes in the label data set 
            output_size = du.CountClasses(labels)
            #Get the test data labels in one hot encoding 
            test_labels = du.ConvertLabels(test_labels, output_size)
            #Get the Labels into a One hot encoding 
            labels = du.ConvertLabels(labels, output_size)

        #Print the labels meta data to the screen and the label data set 
        print("labels:", labels.shape, '\n', labels)
        print("Test labels:", test_labels.shape, '\n', test_labels)
        input_size = X.shape[0]

        ############# hyperparameters ################
        hidden_layers = [input_size]
        learning_rate = .01
        momentum = 0
        batch_size = 20
        epochs = 500
        ##############################################


        NN = NeuralNetwork.NeuralNetwork(
            input_size, hidden_layers, regression, output_size, learning_rate, momentum
        )
        # print("shape x", X.shape)

        # print(vars(NN))
        print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ { data_set } $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        plt.ion()
        batches = batch_input_data(X, labels, batch_size)
        for i in range(epochs):
            
            for batch in batches:
                X_i = batch[0]
                labels_i = batch[1]
                NN.set_input_data(X_i, labels_i)
                NN.forward_pass()
                NN.backpropagation_pass()
            if i % 100 == 0:
                plt.plot(NN.error_x, NN.error_y)
                plt.draw()
                plt.pause(0.00001)
                plt.clf()

        plt.ioff()
        plt.plot(NN.error_x, NN.error_y)
        plt.show()
        print("\n Labels: \n",labels)


        
        Estimation_Values = NN.classify(test_data,test_labels)
        if regression == False: 
            #Decode the One Hot encoding Value 
            Estimation_Values = NN.PickLargest(Estimation_Values)
            test_labels = NN.PickLargest(test_labels)
            print("ESTiMATION VALUES BY GIVEN INDEX (CLASS GUESS) ")
            print(Estimation_Values)
        else: 
            Estimation_Values = Estimation_Values.tolist()
            test_labels = test_labels.tolist() 
            Estimation_Values = Estimation_Values[0]
            test_labels =  test_labels[0]
            #print(test_labels)
            #time.sleep(10000)
        
        Per = Performance.Results()
        Estimat = Estimation_Values
        groun = test_labels
        
        print("ESTIMATE IN LIST FORM")
        print(Estimat)
        print("\n")
        print("GROUND IN LIST FORM ")
        print(groun)

        Nice = Per.ConvertResultsDataStructure(groun, Estimat)
        print("THE GROUND VERSUS ESTIMATION:")
        print(Nice)
        time.sleep(1000000)