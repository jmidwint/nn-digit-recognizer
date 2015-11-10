#!/usr/bin/python

import sys
import code

import math
import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import scipy as sp

# Local files
import sigmoid
import displayData

# Global Variables
coursera = True   # Run the version NN with coursera data

# input_layer_size = 
# NN Architecture Parms 
hidden_layer_size = 25
num_labels = 10 

#========================================
def getCourseraTrainingData():
    ''' Get the data based on the Coursera format
        Returns:
           m: Number of rows
           n: number of features
           X: training examples
           y: labels for X ''' 

    import scipy .io as sio
    fnTrain = '/home/jennym/Kaggle/DigitRecognizer/ex4/ex4data1.mat'
    print("\n Reading in the training file:\n ... " + fnTrain + " ...\n")
    train = sio.loadmat(fnTrain)
    
    X = train['X']
    y = train['y']
    (m, n) = X.shape


    # Replace the 10's with 0 in labels y for indexing purposes
    y[(y == 10)] = 0

    return m, n, X, y

def getKaggleTrainingData():
    ''' Get the data based on the Kaggle format 
        Returns:
           m: Number of rows
           n: number of features
           X: training examples
           y: labels for X ''' 
 
    # Read in and process the input data. 
    fnTrain = '/home/jennym/Kaggle/DigitRecognizer/ex4/train.csv'

    # We can use the pandas library in python to read in the csv file.
    print("\n Reading in the training file:\n ... " + fnTrain + " ...\n")
    train = pd.read_csv(fnTrain)

    # Print the stats of the dataframe.
    print(" Some stats on the data here: \n")
    print(train.describe())

    M = train.as_matrix() # gives numpy array

    X = M[:,1:]    # The training data
    y = M[:, 0:1]  # The labels or answers

    # Scale the input grey scale pixels by 255 to between 0 and 1 float
    X = X.astype('float')
    X = X/X.max()
 
    # Find number of rows examples "m", & columns features "n" 
    m, n = X.shape 

    return m, n, X, y    

# ===========================

#======================


#======================

#===================================================
# Coursera Ex 4 Portion
#  - for validation reasons

def validateFunctions(input_layer_size, X, y ):
    ''' Implements a section of coursera ex4 '''

#%% ================ Part 2: Loading Parameters ================
#% In this part of the exercise, we load some pre-initialized 
# % neural network parameters.

    print('\nLoading Saved Neural Network Parameters ...\n')

    # Load the weights into variables Theta1 and Theta2
    import scipy .io as sio
    fnWeights = '/home/jennym/Kaggle/DigitRecognizer/ex4/ex4weights.mat'
    weights = sio.loadmat(fnWeights)
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    #% Unroll parameters 
    # TODO nn_params = [Theta1[:]  Theta2(:)]
    # nn_params = []

    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

#%% ================ Part 3: Compute Cost (Feedforward) ================
#%  To the neural network, you should first start by implementing the
#%  feedforward part of the neural network that returns the cost only. You
#%  should complete the code in nnCostFunction.m to return cost. After
#%  implementing the feedforward to compute the cost, you can verify that
#%  your implementation is correct by verifying that you get the same cost
#%  as us for the fixed debugging parameters.
#%
#%  We suggest implementing the feedforward cost *without* regularization
#%  first so that it will be easier for you to debug. Later, in part 4, you
#%  will get to implement the regularized cost.
#%
    print('\nFeedforward Using Neural Network ...\n')

    #% Weight regularization parameter (we set this to 0 here).
    MLlambda = 0

    # Cluge, put y back to matlab version, then adjust to use python
    #  indexing later into y_matrix
    y[(y == 0)] = 10
    y = y - 1

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, MLlambda)

    print('Cost at parameters (loaded from ex4weights): ' + str(J) + 
          '\n (this value should be about 0.287629)\n')

    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

#%% =============== Part 4: Implement Regularization ===============
#%  Once your cost function implementation is correct, you should now
#%  continue to implement the regularization with the cost.
#%

    print('\nChecking Cost Function (with Regularization) ... \n')

    # % Weight regularization parameter (we set this to 1 here).
    MLlambda = 1

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, MLlambda)

    print('Cost at parameters (loaded from ex4weights): ' + str(J) +
         '\n(this value should be about 0.383770)\n');

    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  


#%% ================ Part 5: Sigmoid Gradient  ================
#%  Before you start implementing the neural network, you will first
#%  implement the gradient for the sigmoid function. You should complete the
#%  code in the sigmoidGradient.m file.
#%

    print('\nEvaluating sigmoid gradient...\n')

    # TODO g = sigmoidGradient([1 -0.5 0 0.5 1]);
    g = 100 # TODO
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')
    print(g)
    print('\n\n')

    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

 
    return


# ===================================================
#
#                   ===   MAIN   ===
# 
#====================================================

def main():
    ''' Main function - fill this in '''
    #TODO: have all the main calls in here later
    print("\ Hello Jenny, Good Luck with Python.\n")

    ## %% =========== Part 1: Loading and Visualizing Data =============
    #%  We start the exercise by first loading and visualizing the dataset. 
    #%  You will be working with a dataset that contains handwritten digits.
    #%

    if coursera:
        # Read the Courera data
        m, n, X, y = getCourseraTrainingData()
    else:
        # read Kaggle data, and display summary of it.
        m, n, X, y = getKaggleTrainingData()


    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

    # Select some random images from X
    print('Selecting random examples of the data to display.\n')
    sel = np.random.permutation(m)
    sel = sel[0:100]
    images = X[sel, :]

    # Print Out the labels for what is being seen. 
    print('These are the labels for the data ...\n')
    print(y[sel, :].reshape(10, 10))

    # display the sample images
    displayData(images)

    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

    # number of features
    input_layer_size = n    
 
    # validate some functions
    # TODO-move this if statement into the function
    if coursera:
        validateFunctions(input_layer_size, X, y )

    return

# ========================================    
# Go to main if calling from command line
if __name__ == "__main__":
    print("Going to main")
    main()
