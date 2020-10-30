#Written by
#################################################################### MODULE COMMENTS ############################################################################
#THe purpose of this file is to hold small data sets to verify that the program is working as anticipated 
#This program is just a few static functions of predefined data that will lead to calculations known by the programmers 
#There is regression and classification data set values in this program 
#################################################################### MODULE COMMENTS ############################################################################

# module for generating test data for the Neural Network
import numpy as np

class TestData:

    #This will return the array of data and an array of labels as testing for the classification data sets
    @staticmethod
    def single_point_regression():
        #Create a data array 
        data = [
            [.5, .25]
        ]
        #Create a labels array 
        labels = [
            [.66666]
        ]
        #Transpose the arrays and return them 
        return np.array(data).T, np.array(labels).T

    #This will return the array of data and an array of labels as testing for the classification data sets
    @staticmethod
    def regression():
        #Create a data array 
        data = [
            [.5, .25],
            [.56, .22],
            [.33, .20],
            [.36, .18],
            [.34, .88],
            [.28, .94]
        ]
        #Create a labels array 
        labels = [
            [.30],
            [.31],
            [.50],
            [.55],
            [.05],
            [.04]
        ]
        #Transpose the arrays and return them 
        return np.array(data).T, np.array(labels).T

    #This will return the array of data and an array of labels as testing for the classification data sets
    @staticmethod
    def single_point_classification():
        #Create a data array 
        data = [
            [.5, .25]
        ]
    #Create a labels array 
        labels = [
            [1]
        ]
        #Transpose the arrays and return them 
        return np.array(data).T, np.array(labels).T

    #This will return the array of data and an array of labels as testing for the classification data sets
    @staticmethod
    def classification():
        #Create a data array 
        data = [
            [.5, .25],
            [.56, .22],
            [.33, .20],
            [.36, .18],
            [.34, .88],
            [.28, .94]
        ]
        #Create a one hot encoded labels array 
        labels = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1]
        ]
        #Transpose the arrays and return them 
        return np.array(data).T, np.array(labels).T