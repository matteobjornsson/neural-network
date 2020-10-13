#Written by Nick Stone and edited by Matteo Bjornsson 
##################################################################### MODULE COMMENTS ####################################################################
# The following Python object is responsible for calculating two loss functions to identify a series of statistical data points for a programmer to view #
# In order to see how 'Well' the Naive bayes program is functioning. The two loss functions that Nick Stone and Matteo Bjornsson implemented for this pr-#
# -oject were the 0/1 loss function which we will use to calculate the algorithms precision and the F1 score for a multidimensional data set.            #
# All of the functions have been documented such that a programmer can understand the mathematics and statistics involved for undersanding each of the l-#
# -oss Functions. The main datastructures used were a dataframe and a dictionary to keep track of a given confusion matrix                               # 
#################################################################### MODULE COMMENTS ####################################################################
import pandas as pd
import numpy as np




class Results: 
    
    """
    loss functions 

    multiclass confusion matrix 
    https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

    multiclass precision and recall
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

    multiclass f1 score
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

    """
    def LossFunctionPerformance(self,Regression,Datalist):
        #Create a list to hold data points to be written to a file  
        DataPackage = list() 
        #The data set is categorical in value run F1 and Zero one loss functions 
        if Regression == False:
            #Store the Zero/One loss function values
            Zero = self.ZeroOneLoss(Datalist)
            #Run the 0/1 Loss function and F1 SCore and store the value 
            F1 = self.statsSummary(Datalist)
            F1 = F1 * 100 
            DataPackage.append(F1)
            DataPackage.append(Zero)
        #The value that is being tested is regression value  
        else:
            #Run Mean Absolute Error and store the value to piped to a file  
            MAE = self.MAE(Datalist)
            #Run The mean squared error and store the value to be piped to a file  
            MSE  = self.MSE(Datalist)
            DataPackage.append(MAE)
            DataPackage.append(MSE)
        #Print all of the data generated in the loss functions to a csv file for programmer review 
        return DataPackage

    def CrossEntropy(targets, predictions): 
        print(targets)
    
    def StartLossFunction(self,Regression,Datalist,MetaData, filename='experimental_results.csv'):
        #Create a list to hold data points to be written to a file  
        DataPackage = list() 
        #The data set is categorical in value run F1 and Zero one loss functions 
        if Regression == False: 
            #Store the Zero/One loss function values
            Zero = self.ZeroOneLoss(Datalist)
            #Run the 0/1 Loss function and F1 SCore and store the value 
            F1 = self.statsSummary(Datalist)
            F1 = F1 * 100 
            # DataPackage.append("Zero One ")
            DataPackage.append(Zero)
            # DataPackage.append("F1 Score")
            DataPackage.append(F1)
        #The value that is being tested is regression value  
        else:
            #Run Mean Absolute Error and store the value to piped to a file  
            MAE = self.MAE(Datalist)
            #Run The mean squared error and store the value to be piped to a file  
            MSE  = self.MSE(Datalist)
            # DataPackage.append("Mean Absolute Error")
            DataPackage.append(MAE)
            # DataPackage.append("Mean Squared Error")
            DataPackage.append(MSE)
        #Print all of the data generated in the loss functions to a csv file for programmer review 
        self.PipeToFile(DataPackage, MetaData, filename)
        return DataPackage

    def PipeToFile(self,DataPackage,MetaData, filename): 
        #Try to access the file that we are trying to write too 
        try: 
            #Open the CSV file in append mode to be written to 
            with open(filename ,mode = "a") as file: 
                count = 0 
                #For each of the data points stored in the metadata 
                for i in MetaData: 
                    if count == len(MetaData): 
                        file.write(str(i))
                        continue 
                    print(i)
                    #Write a given input into a row in the file 
                    file.write(str(i) + ',')
                    count += 1 
                count = 0 
                #For each of the loss functions calculated (2)
                for j in DataPackage: 
                    count += 1 
                    if count == len(DataPackage): 
                        file.write(str(j))
                        continue 
                    #Write the loss function data to the file 
                    file.write(str(j) + ',')
                    
                file.write("\n")
                file.close() 
        #If we cannot print a message to the screen 
        except: 
            #Print some output to the user so they can check whether the file is in use 
            print("An Error Occured Trying to read the File KNNResults.csv")

    #Parameters: List of data set 
    #Returns: the float for the mean absolute error 
    #Function: Take in a dataframe and count the number of correct classifications and return the percentage value 
    def MAE(self,Data_set: list())-> float: 
        #Create an absolute value list
        MeanAbs = list() 
        #For each of the lists in the data setpassed in 
        for i in Data_set: 
            #Store the true value
            True_Value = i[0]
            #Store the predicted value 
            Predict_Value = i[1]
            #Store the absolute value of the difference of the above values
            absolute = abs(True_Value - Predict_Value)
            #Store the absolute value in the list 
            MeanAbs.append(absolute)
        #SEt a mean variable to be 0 
        mean = 0 
        #For each of the absolute values stored 
        for i in MeanAbs: 
            #Add the value to the variable 
            mean += i
        #Generate the mean from the list 
        mean = mean / (len(MeanAbs)+ .0000000001)
        #Return the mean 
        return mean 

    def MSE(self,data_set: list()) -> float: 
        SquaredError = list()  
        for i in data_set: 
            #First Value is the Ground truth 
            True_Value = i[0]
            #Grab the last value since it is the predicted value 
            Pred_Value = i[1]
            #Calculate the error by the difference of the two values above 
            Error = True_Value - Pred_Value
            #Square the error 
            Error = Error * Error
            #Store into the Squared Error list created above 
            SquaredError.append(Error)
        #Set a counter variable 
        Mean = 0 
        #For each of the squared error vales we entered in the list above 
        for i in SquaredError: 
            #Add the value to the overall mean 
            Mean +=i 
        #Divide out by the total number of entries 
        Mean = Mean / (len(SquaredError)+ .0000000001)
        #Return the mean 
        return Mean 
         

   
####################################### UNIT TESTING #################################################
if __name__ == '__main__':
    print("Program Start")
    print("Program Finish")



####################################### UNIT TESTING #################################################
    



