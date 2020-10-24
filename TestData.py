# module for generating test data for the Neural Network
import numpy as np

class TestData:

    @staticmethod
    def single_point_regression():
        data = [
            [.5, .25]
        ]
        labels = [
            [.66666]
        ]
        return np.array(data).T, np.array(labels).T

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
            [.30],
            [.31],
            [.50],
            [.55],
            [.05],
            [.04]
        ]
        return np.array(data).T, np.array(labels).T

    @staticmethod
    def single_point_classification():
        data = [
            [.5, .25]
        ]
        labels = [
            [1]
        ]
        return np.array(data).T, np.array(labels).T

    @staticmethod
    def classification():
        data = [
            [.5, .25],
            [.56, .22],
            [.33, .20],
            [.36, .18],
            [.34, .88],
            [.28, .94]
        ]
        labels = [
            [1],
            [1],
            [2],
            [2],
            [3],
            [3]
        ]
        return np.array(data).T, np.array(labels).T