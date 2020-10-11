# module for generating test data for the Neural Network
import numpy as np

class TestData:

    @staticmethod
    def regression():
        data = [
            [.5, .25],
            [.56, .22],
            [.33, .20],
            [.36, .18],
            [.34, .88],
            [.28, .94]
        ]
        labels = [
            [30],
            [31],
            [50],
            [55],
            [5],
            [4]
        ]
        return np.array(data), np.array(labels)

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
        labels = [
            [1],
            [1],
            [2],
            [2],
            [3],
            [3]
        ]
        return np.array(data), np.array(labels)