# Parrallel driver conversion written by Matteo Bjornsson
################################################################################
# This module is the same as the Driver.py file but dispatches each trial as a
# asynch job to take advantage of all cpu threads. Like the Driver.py file, it
# runs the entire experiment. Data is prepped into 10 folds, and a 
# neural network is trained and tested on each fold of each data set for 0, 1,
# and 2 hidden layers. The performance of each run is written to file. 
################################################################################
import numpy as np
import pandas as pd
import math, random, copy
import TestData
import DataUtility
import NeuralNetwork
import matplotlib.pyplot as plt
import time 
import Performance
import multiprocessing

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

# this function is responsible for training the model, estimating the test data,
# and recording the results, given all the specific parameters for the data set/model
def driver(q, input_size_d, hidden_layers_d, regression_d, output_size_d, learning_rate_d, momentum_d,
            X_d, labels_d, batch_size_d, epochs_d, test_data_d, test_labels_d, status_print, data_set):
    # initialize a NN given the input parameters
    NN = NeuralNetwork.NeuralNetwork(
        input_size_d, hidden_layers_d, regression_d, output_size_d, learning_rate_d, momentum_d
    )
    # batch the training data so that there is a forward pass for each batch per epoch
    batches = batch_input_data(X_d, labels_d, batch_size_d)
    # run the forward pass and backprop for the perscribed number of epochs
    for i in range(epochs_d):
        for batch in batches:
            X_i = batch[0]
            labels_i = batch[1]
            NN.set_input_data(X_i, labels_i)
            NN.forward_pass()
            NN.backpropagation_pass()
    # estimate the output values of the test data
    Estimation_Values = NN.classify(test_data_d,test_labels_d)
    #depending if it is regression or classification, translate the output to a class estimate
    if regression_d == False: 
        #Decode the One Hot encoding Value 
        Estimation_Values = NN.PickLargest(Estimation_Values)
        test_labels_list = NN.PickLargest(test_labels_d)
    else: 
        Estimation_Values = Estimation_Values.tolist()
        test_labels_list = test_labels_d.tolist()[0]
        Estimation_Values = Estimation_Values[0]
    
    results = Per.ConvertResultsDataStructure(test_labels_list, Estimation_Values)

    #Meta Data order
    h1 = 0 
    h2 = 0 
    #The number of hidden layers is 0 
    if len(hidden_layers_d) == 0: 
        #No hidden layers so 0 
        h1 = 0 
        h2 = 0 
    #THe number of hidden layers is 1 
    elif len(hidden_layers_d) == 1: 
        #Set the number of nodes in the hidden layer 
        h1 = hidden_layers_d[0]
        #No layer so 0
        h2 = 0 
    #The number of hidden layers is 2 
    else: 
        #The number of nodes per hidden layer 
        h1 = hidden_layers_d[0]
        #The number of nodes per hidden layer 
        h2 = hidden_layers_d[1]

    #[data set, number of h layers, node per h 1, nodes per h2, learning rate, momentum, batch size, number batches, number epochs]
    Meta = [data_set, len(hidden_layers_d), h1, h2, learning_rate_d, momentum_d, batch_size_d, len(batches), epochs_d]
    results_performance = Per.LossFunctionPerformance(regression_d,results)
    # prep the results to a string that will be written to file
    data_point = Meta + results_performance
    data_point_string = ','.join([str(x) for x in data_point])
    # put the result on the multiprocessing queue
    q.put(data_point_string)
    # print out that the job is done and progress i.e. 5/20 Abalone jobs
    print(status_print)

# this function takes the results from the queue that all async jobs write to, and
# writes the jobs to disk. This function is meant to be started as it's own process.
# param q is the multiprocess Manager queue object shared by all jobs. 
def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

###############################################################################
#               START EXPERIMENT:                                             #
###############################################################################
data_sets = ["soybean", "glass", "abalone","Cancer","forestfires", "machine"] 

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
headers = ["Data set", "Hidden Layers", "h1 nodes", "h2 nodes", "learning rate", "momentum", "batch size", "batches", "epochs", "loss1", "loss2"]
filename = 'experimental_results.csv'

Per = Performance.Results()
Per.PipeToFile([], headers, filename)

#########################################################
################ TUNED HYPERPARAMETERS ##################
#########################################################
tuned_0_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 5000,
        "hidden_layer": []
    },
    "Cancer": {
        "learning_rate": .00001,
        "batch_count": 20,
        "epoch": 10000,
        "hidden_layer": []
    },
    "glass": {
        "learning_rate": .1,
        "batch_count": 10,
        "epoch": 50000,
        "hidden_layer": []
    },
    "forestfires": {
        "learning_rate": .00001,
        "batch_count": 10,
        "epoch": 10000,
        "hidden_layer": []
    },
    "machine": {
        "learning_rate": .1,
        "batch_count": 5,
        "epoch": 10000,
        "hidden_layer": []
    },
    "abalone": {
        "learning_rate": .1,
        "batch_count": 10,
        "epoch": 10000,
        "hidden_layer": []
    }
}

tuned_1_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 10,
        "epoch": 10000,
        "hidden_layer": [7]
    },
    "Cancer": {
        "learning_rate": .000001,
        "batch_count": 5,
        "epoch": 100000,
        "hidden_layer": [4]
    },
    "glass": {
        "learning_rate": .001,
        "batch_count": 10,
        "epoch": 10000,
        "hidden_layer": [8]
    },
    "forestfires": {
        "learning_rate": .00001,
        "batch_count": 5,
        "epoch": 50000,
        "hidden_layer": [8]
    },
    "machine": {
        "learning_rate": .001,
        "batch_count": 50,
        "epoch": 50000,
        "hidden_layer": [4]
    },
    "abalone": {
        "learning_rate": .01,
        "batch_count": 10,
        "epoch": 50000,
        "hidden_layer": [8]
    }
}

tuned_2_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 50000,
        "hidden_layer": [7,12]
    },
    "Cancer": {
        "learning_rate": .0000001,
        "batch_count": 5,
        "epoch": 100000,
        "hidden_layer": [4,4]
    },
    "glass": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 10000,
        "hidden_layer": [8,6]
    },
    "forestfires": {
        "learning_rate": .0001,
        "batch_count": 10,
        "epoch": 50000,
        "hidden_layer": [8,8]
    },
    "machine": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 50000,
        "hidden_layer": [7,2]
    },
    "abalone": {
        "learning_rate": .01,
        "batch_count": 10,
        "epoch": 50000,
        "hidden_layer": [6,8]
    }
}

##############################################
# START MULTIPROCESS JOB POOL
##############################################
manager = multiprocessing.Manager()
q = manager.Queue()
start = time.time()
writer = multiprocessing.Process(target=data_writer, args=(q,filename))
writer.start()

pool = multiprocessing.Pool()

#################################################
# RUN THE NN FOR EACH
#   DATA SET, FOLD, AND NUMBER OF HIDDEN LAYERS 
#################################################

for data_set in data_sets:
    counter = 1
    for j in range(10):
        
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)
        test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[j])
        #Append all data folds to the training data set
        remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
        remaining_labels = [y[1] for i, y in enumerate(tenfold_data_and_labels) if i!=j]
        #Store off a set of the remaining dataset 
        X = np.concatenate(remaining_data, axis=1) 
        #Store the remaining data set labels 
        labels = np.concatenate(remaining_labels, axis=1)
        print(data_set, "training data prepared")
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
        input_size = X.shape[0]

        data_set_size = X.shape[1] + test_data.shape[1]
        momentum = 0
        
        tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]
        for i in range(3):

            # collect the tuned parameters depending on how many hidden layers
            learning_rate = tuned_parameters[i]["learning_rate"]
            batch_count = tuned_parameters[i]["batch_count"]
            epoch = tuned_parameters[i]["epoch"]
            hidden_layers = tuned_parameters[i]["hidden_layer"]

            #printout to determine how far along each data set is
            status_print = f"Data Set: {data_set} {counter}/30"

            # Start a job with the given data set, fold, and hidden layers
            pool.apply_async(driver, args=(
                q, 
                input_size,
                hidden_layers,
                regression,
                output_size,
                learning_rate,
                momentum,
                X,
                labels,
                batch_count,
                epoch,
                test_data,
                test_labels,
                status_print,
                data_set
                )
            )
            counter += 1
##############################
# CLOSE THE MULTIPROCESS POOL
##############################
pool.close()
pool.join()
q.put('kill')
writer.join()
elapsed_time = time.time() - start
print("Elapsed time: ", elapsed_time, 's')
