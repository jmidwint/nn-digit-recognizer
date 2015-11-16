#!/usr/bin/python

import sys
import code

import math
import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import scipy as sp
import scipy.optimize as sci

# Local files
#print sys.path
from Stage2.getKaggleTrainingData import getKaggleTrainingData
from Stage1.displayData import displayData
from Stage1.randInitializeWeights import randInitializeWeights
from Stage1.nnCostFunction import nnCostFunction
#from sigmoidGradient import sigmoidGradient
#from randInitializeWeights import randInitializeWeights
#from checkNNGradients import checkNNGradients
from Stage1.pred import predict
from Stage2.obtainKaggleTestResults import obtainKaggleTestResults, writeKaggleTestResults
from Stage2.learnedTheta import writeLearnedTheta



# Global Variables
coursera = True   # Run the version NN with coursera data

# input_layer_size = 
# NN Architecture Parms 
hidden_layer_size = 100 # from 25
num_labels = 10 
MAXITER = 1000 # from 50, 400


# ===================================================
#
#                   ===   MAIN   ===
# 
#====================================================

def main():
    ''' Main function - fill this in '''

    ## %% =========== Part 1: Loading and Visualizing Data =============
    #%  We start the exercise by first loading and visualizing the dataset. 
    #%  You will be working with a dataset that contains handwritten digits.
    #%

 
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

    # display the sample images
    displayData(images)

    # Print Out the labels for what is being seen. 
    print('These are the labels for the data ...\n')
    print(y[sel, :].reshape(10, 10))

    # number of features
    input_layer_size = n    
    # debug - jkm
    print sys.path


    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  


##  ================ Skipping Parts 2 to 5 ==========================

#%% ================ Part 6: Initializing Parameters ================
#%  In this part of the exercise, you will be starting to implement a two
#%  layer neural network that classifies digits. You will start by
#%  implementing a function to initialize the weights of the neural network
#%  (randInitializeWeights.m)

    print('\nInitializing Neural Network Parameters ...\n')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    #% Unroll parameters
    initial_nn_params = np.hstack(( initial_Theta1.ravel(order = 'F'),
                                   initial_Theta2.ravel(order = 'F')))
    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  


##  ================ Skipping Parts 7 , 8  ==========================

#%% =================== Part 8b: Training NN ===================
#%  You have now implemented all the code necessary to train a neural 
#%  network. To train your neural network, we will now use "fmincg", which
#%  is a function which works similarly to "fminunc". Recall that these
#%  advanced optimizers are able to train our cost functions efficiently as
#%  long as we provide them with the gradient computations.
#%
    #%  After you have completed the assignment, change the MaxIter to a larger
    #%  value to see how more training helps.
    #% jkm change maxIter from 50-> 400
    options = {'maxiter': MAXITER}

    #%  You should also try different values of lambda
    MLlambda = 1

    print ('\nTraining Neural Network... \n')
    print('\n  Parms: Hidden Layer Units: {0}  Max Iters: {1}  Lambda: {2}  \n'.format( 
                hidden_layer_size, MAXITER, MLlambda))
   
    #% Create "short hand" for the cost function to be minimized
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                               num_labels, X, y, MLlambda)

    #% Now, costFunction is a function that takes in only one argument (the
    #% neural network parameters)

    '''
    NOTES: Call scipy optimize minimize function
        method : str or callable, optional Type of solver. 
           CG -> Minimization of scalar function of one or more variables 
                 using the conjugate gradient algorithm.

        jac : bool or callable, optional Jacobian (gradient) of objective function. 
              Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. 
              If jac is a Boolean and is True, fun is assumed to return the gradient 
              along with the objective function. If False, the gradient will be 
              estimated numerically. jac can also be a callable returning the 
              gradient of the objective. In this case, it must accept the same 
              arguments as fun.
        callback : callable, optional. Called after each iteration, as callback(xk), 
              where xk is the current parameter vector.
'''
    # Setup a callback for displaying the cost at the end of each iteration 
    class Callback(object): 
        def __init__(self): 
            self.it = 0 
        def __call__(self, p): 
            self.it += 1 
            print "Iteration %5d | Cost: %e" % (self.it, costFunc(p)[0]) 
 
   
    result = sci.minimize(costFunc, initial_nn_params, method='CG', 
                   jac=True, options=options, callback=Callback()) 
    nn_params = result.x 
    cost = result.fun 
 
    # matlab: [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    #% Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
               (hidden_layer_size, (input_layer_size + 1)), 
                order = 'F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
               (num_labels, (hidden_layer_size + 1)), 
               order = 'F')  


    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")


#%% ================= Part 9: Visualize Weights =================
#%  You can now "visualize" what the neural network is learning by 
#%  displaying the hidden units to see what features they are capturing in 
#%  the data.#

    print('\nVisualizing Trained Neural Network... \n')
    print('\n  Parms: Hidden Layer Units: {0}  Max Iters: {1}  Lambda: {2}  \n'.format( 
                hidden_layer_size, MAXITER, MLlambda))

    displayData(Theta1[:, 1:])

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")


#%% ================= Part 10: Implement Predict =================
#%  After training the neural network, we would like to use it to predict
#%  the labels. You will now implement the "predict" function to use the
#%  neural network to predict the labels of the training set. This lets
#%  you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    # print out what the images are predicted to be
    print('Selecting random examples of the data to display and how they are predicted.\n')
    print(pred[sel].reshape(10, 10))

    # display the sample images
    displayData(images)

    # JKM - my array was column stacked - don't understand why this works
    pp = np.row_stack(pred)
    accuracy = np.mean(np.double(pp == y)) * 100

    print('\n Training Set Accuracy: {0} \n'.format(accuracy))
    print('\n  Parms: Hidden Layer Units: {0}  Max Iters: {1}  Lambda: {2}  \n'.format( 
                hidden_layer_size, MAXITER, MLlambda))
    

    # Create a filname to use to write the results
    fn = 'HU_{0}_MaxIter_{1}_Lambda_{2}_PredAcc_{3}'.format(
           hidden_layer_size, MAXITER, MLlambda, accuracy)

    # Capture the Thetas
    writeLearnedTheta(Theta1, 'Theta1_' + fn)
    writeLearnedTheta(Theta2, 'Theta2_' + fn)
 

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

#%% ================= Part 11: Find Best Results  =================
#      JKM - I will loop through and find the best results
#         and store the NN 
#       but for now just continue
#

    

#%% ================= Part 12: Run predictionon Test data  =================
#      Run teh prediction on teh test data 
#      write the results to a csv file to submit to kaggle 
#

    ## NEXT PART
    kagglePred = obtainKaggleTestResults(Theta1, Theta2)    
    writeKaggleTestResults(fn, kagglePred)

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    return

# ========================================    
# Go to main if calling from command line
if __name__ == "__main__":
    print("Going to main")
    main()
