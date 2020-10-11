# module for generating test data for the Neural Network
import numpy as np

class TestData:
    
    @staticmethod
    def regression():
        data = [
            [.5, .25, 30],
            [.56, .22, 31],
            [.33, .20, 50],
            [.36, .18, 55],
            [.34, .88, 5],
            [.28, .94, 4]
        ]
        return np.array(data)

    @staticmethod
    def classification():
        data = [
            [.5, .25, 1],
            [.56, .22, 1],
            [.33, .20, 2],
            [.36, .18, 2],
            [.34, .88, 3],
            [.28, .94, 3]
        ]
        return np.array(data)