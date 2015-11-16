import numpy as np

def sigmoid(z):
    ''' SIGMOID Compute sigmoid function
    ''' 
    g = 1 / (1 + np.exp(-z))
    return g
