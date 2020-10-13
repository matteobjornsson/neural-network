
import numpy as np 




class VectorHelper: 
    

    def __inti__(self): 
        print("Vector Helper has been created")


    def SoftMax(a): 
        soft = np.exp(a)
        soft = soft/soft.sum()
        return soft

    def CrossEntropy(a): 
        print(a)
    

    def Sigmoid(a):
        print(a)


    def SignmoidDerivative(a): 
        print(a)

    def DotProduct(V1, V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] * V2[i])
        return NewDotVector


    def VectorAddition(V1,V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] + V2[i])
        return NewDotVector

    def VectorSubtraction(V1,V2): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] - V2[i])
        return NewDotVector

    def VectorScaling(V1,Scalar): 
        NewDotVector= list() 
        for i in range(len(V1)): 
            NewDotVector.append(v1[i] * Scalar)
        return NewDotVector