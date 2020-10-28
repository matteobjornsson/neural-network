
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

def driver(q, input_size_d, hidden_layers_d, regression_d, output_size_d, learning_rate_d, momentum_d,
            X_d, labels_d, batch_size_d, epochs_d, test_data_d, test_labels_d, status_print, data_set):
    # print(data_set, "job started. Epochs:", epochs_d, "Layers:", len(hidden_layers), "Learning rate:", learning_rate_d, "Batch size:", batch_size_d)

    NN = NeuralNetwork.NeuralNetwork(
        input_size_d, hidden_layers_d, regression_d, output_size_d, learning_rate_d, momentum_d
    )
    # print("shape x", X.shape)

    # print(vars(NN))
    # print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ { data_set } $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    # print("total cycles: ", math.ceil(X_d.shape[1]/batch_size_d)*epochs_d, "batches:",int(X_d.shape[1]/batch_size_d), "batch size:", batch_size_d, "learning rate:", learning_rate_d)
    batches = batch_input_data(X_d, labels_d, batch_size_d)
    for i in range(epochs_d):
        for batch in batches:
            X_i = batch[0]
            labels_i = batch[1]
            NN.set_input_data(X_i, labels_i)
            NN.forward_pass()
            NN.backpropagation_pass()

        # if i % 100 == 0:
        #     if counter < 2:
        #         plt.plot(NN.error_x, NN.error_y)
        #         plt.draw()
        #         plt.pause(0.00001)
        #         plt.clf()
    # if counter == 1:
    #     plt.ioff()
    #     plt.plot(NN.error_x, NN.error_y)
    #     plt.show()
    #     print("\n Labels: \n",labels)

    Estimation_Values = NN.classify(test_data_d,test_labels_d)
    if regression_d == False: 
        #Decode the One Hot encoding Value 
        Estimation_Values = NN.PickLargest(Estimation_Values)
        test_labels_list = NN.PickLargest(test_labels_d)
        # print("ESTiMATION VALUES BY GIVEN INDEX (CLASS GUESS) ")
        # print(Estimation_Values)
    else: 
        Estimation_Values = Estimation_Values.tolist()
        test_labels_list = test_labels_d.tolist()[0]
        Estimation_Values = Estimation_Values[0]

        #print(test_labels)
        #time.sleep(10000)
    
    Estimat = Estimation_Values
    groun = test_labels_list
    
    # print("ESTIMATE IN LIST FORM")
    # print(Estimat)
    # print("\n")
    # print("GROUND IN LIST FORM ")
    # print(groun)

    Nice = Per.ConvertResultsDataStructure(groun, Estimat)
    # print("THE GROUND VERSUS ESTIMATION:")
    # print(Nice)
    """
    hidden_layers = [input_size]
    learning_rate = .01
    momentum = 0
    batch_size = 20
    epochs = 500
    """
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
    results_set = Per.LossFunctionPerformance(regression_d,Nice)
    data_point = Meta + results_set
    data_point_string = ','.join([str(x) for x in data_point])
    # put the result on the multiprocessing queue
    q.put(data_point_string)
    print(status_print)

def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

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
filename = 'nodes_per_hidden_layer_tuning.csv'

Per = Performance.Results()
Per.PipeToFile([], headers, filename)

tuned_0_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 5000
    },
    "Cancer": {
        "learning_rate": .00001,
        "batch_count": 20,
        "epoch": 10000
    },
    "glass": {
        "learning_rate": .1,
        "batch_count": 10,
        "epoch": 50000
    },
    "forestfires": {
        "learning_rate": .00001,
        "batch_count": 10,
        "epoch": 10000
    },
    "machine": {
        "learning_rate": .1,
        "batch_count": 5,
        "epoch": 10000
    },
    "abalone": {
        "learning_rate": .1,
        "batch_count": 10,
        "epoch": 10000
    }
}

tuned_1_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 10,
        "epoch": 10000
    },
    "Cancer": {
        "learning_rate": .000001,
        "batch_count": 5,
        "epoch": 100000
    },
    "glass": {
        "learning_rate": .001,
        "batch_count": 10,
        "epoch": 10000
    },
    "forestfires": {
        "learning_rate": .00001,
        "batch_count": 5,
        "epoch": 50000
    },
    "machine": {
        "learning_rate": .001,
        "batch_count": 50,
        "epoch": 50000
    },
    "abalone": {
        "learning_rate": .01,
        "batch_count": 10,
        "epoch": 50000
    }
}

tuned_2_hl = {
    "soybean": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 50000
    },
    "Cancer": {
        "learning_rate": .0000001,
        "batch_count": 5,
        "epoch": 100000
    },
    "glass": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 10000
    },
    "forestfires": {
        "learning_rate": .0001,
        "batch_count": 10,
        "epoch": 50000
    },
    "machine": {
        "learning_rate": .001,
        "batch_count": 5,
        "epoch": 50000
    },
    "abalone": {
        "learning_rate": .01,
        "batch_count": 10,
        "epoch": 50000
    }
}

manager = multiprocessing.Manager()
q = manager.Queue()
start = time.time()
writer = multiprocessing.Process(target=data_writer, args=(q,filename))
writer.start()

pool = multiprocessing.Pool()

data_set_counter = 1
for data_set in data_sets:
    
    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    # ten fold data and labels is a list of [data, labels] pairs, where 
    # data and labels are numpy arrays:
    tenfold_data_and_labels = du.Dataset_and_Labels(data_set)
    test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[0])
    #Append all data folds to the training data set
    remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=0]
    remaining_labels = [y[1] for i, y in enumerate(tenfold_data_and_labels) if i!=0]
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

    tuning_h1 = [m for m in reversed(range(2, input_size + 1, int(input_size/4)))]
    tuned_h1_parameters = tuned_1_hl[data_set]
    lr_h1 = tuned_h1_parameters["learning_rate"]
    bc_h1 = tuned_h1_parameters["batch_count"]
    e_h1 = tuned_h1_parameters["epoch"]

    tuning_h2 = [n for n in reversed(range(2, input_size + 1, int(input_size/4)))]
    tuned_h2_parameters = tuned_2_hl[data_set]
    lr_h2 = tuned_h2_parameters["learning_rate"]
    bc_h2 = tuned_h2_parameters["batch_count"]
    e_h2 = tuned_h2_parameters["epoch"]

    counter = 1
    total = len(tuning_h1) + len(tuning_h1)**2

    for h1 in tuning_h1:
        h = [h1]
        status_print = f"Data Set: {data_set} {counter}/{total}"

        pool.apply_async(driver, args=(
            q, 
            input_size,
            h,
            regression,
            output_size,
            lr_h1,
            momentum,
            X,
            labels,
            bc_h1,
            e_h1,
            test_data,
            test_labels,
            status_print,
            data_set
            )
        )
        counter += 1

        for h2 in tuning_h2:
            h = [h1, h2]
            status_print = f"Data Set: {data_set} {counter}/{total}"

            pool.apply_async(driver, args=(
                q, 
                input_size,
                h,
                regression,
                output_size,
                lr_h2,
                momentum,
                X,
                labels,
                bc_h2,
                e_h2,
                test_data,
                test_labels,
                status_print,
                data_set
                )
            )
            counter += 1
    
pool.close()
pool.join()
q.put('kill')
writer.join()
elapsed_time = time.time() - start
print("Elapsed time: ", elapsed_time, 's')
