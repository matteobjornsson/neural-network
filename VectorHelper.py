
import numpy as np 




class VectorHelper: 
    

    def __init__(self): 
        print("Vector Helper has been created")


    def SoftMax(self,a): 
        soft = np.exp(a)
        soft = soft/soft.sum()
        return soft

    def Sigmoid(self,a):
        return 1/(1+np.exp(-a))  


    def SignmoidDerivative(self,a): 
        return Sigmoid(a) * (1-Sigmoid(a))

    def DotProduct(self,V1, V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] * V2[i])
        return NewDotVector


    def VectorAddition(self,V1,V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] + V2[i])
        return NewDotVector

    def VectorSubtraction(self,V1,V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] - V2[i])
        return NewDotVector

    def VectorScaling(self,V1,Scalar): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] * Scalar)
        return NewDotVector



