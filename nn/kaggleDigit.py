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
from Stage2.getKaggleTrainingData import getKaggleTrainingData, displaySampleData, partitionData
from Stage1.displayData import displayData
from Stage1.nnCostFunction import nnCostFunction
#from sigmoidGradient import sigmoidGradient
from Stage1.randInitializeWeights import randInitializeWeights
from Stage1.checkNNGradients import checkNNGradients
from Stage1.pred import predict
from Stage2.obtainKaggleTestResults import obtainKaggleTestResults, writeKaggleTestResults
from Stage2.learnedTheta import writeLearnedTheta
from Stage2.trainNN import trainNN, visualizeNN



# Global Variables
coursera = True   # Run the version NN with coursera data

# input_layer_size = 
# NN Architecture Parms 
hidden_layer_size = 25 # from 25
num_labels = 10 
MAXITER = 400 # from 50, 400


# ===================================================
#
#                   ===   MAIN   ===
# 
#====================================================

def main():
    ''' Main function - fill this in '''

    ## %% =========== Part 1: Loading and Visualizing Data =============
    #%
 
    # read Kaggle data, and display summary of it.
    m, n, X, y = getKaggleTrainingData()

    # display a sample of the data & corresponding labels
    displaySampleData(X, y)   

    # Partition the Kaggle training data
    X_train, y_train, X_cv, y_cv, X_test, y_test =  partitionData(X, y)

    # jkm - debug
    print '\n Kaggle Data Partitioned into Train, cv, test'
    print 'Labels :', np.arange(10)
    print 'y      :', np.bincount(np.hstack(y))  
    print 'y_train:', np.bincount(np.hstack(y_train))
    print 'y_cv   :', np.bincount(np.hstack(y_cv))
    print 'y_test :', np.bincount(np.hstack(y_test))
    print 'Totals :', np.bincount(np.hstack(y_train)) + np.bincount(np.hstack(y_cv)) + np.bincount(np.hstack(y_test)) 

    # features
    input_layer_size = n    

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

#%% ================= Part 2: Train the NN  =================
#      JKM - I will loop through and find the best results
#         and store the NN 
#       but for now just continue
#



    # jkmm - put inline again
    # Callback for displaying the cost at the end of each iteration 
    class Callback(object): 
        def __init__(self, input_layer_size, hidden_layer_size, num_labels,
                            X_cv, y_cv, _lambda, costFunc): 
            self.it = 0
            self.input_layer_size = input_layer_size
            self.hidden_layer_size = hidden_layer_size
            self.num_labels = num_labels
            self.X_cv = X_cv
            self.y_cv = y_cv
            self._lambda = _lambda
            self.costFunc = costFunc

        def __call__(self, p ): 
            self.it += 1
            J_train = self.costFunc(p)[0]

            # Calculate the cv cost every 10 iterations
            if (self.it % 10 == 0):
                J_cv, _ = nnCostFunction(p, self.input_layer_size, self.hidden_layer_size,
                                   self.num_labels, self.X_cv, self.y_cv, self._lambda)

                diff = np.abs(J_train - J_cv) 

                print "Iter %5d | J_train: %e  | J_cv: %e  | Diff: %e" % (self.it, J_train, J_cv, diff) 
            else:
                print "Iter %5d | J_train: %e" % (self.it, J_train)


    ''' put back the code into inline on this script
        for some speed improvement
    Theta1, Theta2 = trainNN(input_layer_size, hidden_layer_size, num_labels, _lambda, 
             X_train, y_train, X_cv, y_cv)
 
    '''


    # Step 1: Initializing  Parameters
    # initial_nn_params = initializeNN(input_layer_size, hidden_layer_size, num_labels)
    # make inline again 

    print('\nInitializing Neural Network Parameters ...\n')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    #% Unroll parameters
    initial_nn_params = np.hstack(( initial_Theta1.ravel(order = 'F'),
                                    initial_Theta2.ravel(order = 'F')))

 
 
    options = {'maxiter': MAXITER} # jkm - need to think about finding this best value
    best_lambda = np.core.numeric.NaN
    best_acc = 0
   
    ########### Start of Loop #########
    # Loop to find best _lambda, using train data to tarin, and cv data to evaluate.
    # started this loop , stopped and restarted to make lambda a float [0.1, 0.5,  1, 3, 5]
    for _lambda in [1.0,  2.0, 3.0, 5.0]:  # this is to be calculated later
        ## jkm - need to make lambdas as floats. 
        #  actually _lambda gets turned into a float in teh cost function anyway.
        _lambda = float(_lambda)
        # Step 2: Training NN 
        print ('\nTraining Neural Network... \n')
        print('\n  Parms: Hidden Layer Units: {0}  Max Iters: {1}  Lambda: {2}  \n'.format( 
                    hidden_layer_size, MAXITER, _lambda))

        #% Create "short hand" for the cost function to be minimized
        #% Now, costFunction is a function that takes in only one argument (the
        #% neural network parameters)

        costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                   num_labels, X_train, y_train, _lambda)

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

        result = sci.minimize(costFunc, initial_nn_params, method='CG', 
                   jac=True, options=options, 
                   callback=Callback(input_layer_size, hidden_layer_size, 
                                     num_labels, X_cv, y_cv, _lambda, costFunc)) 
        nn_params = result.x 
        cost = result.fun 

        # Debug statement
        print('\n Results from minimizer function Success: {0} \n   {1} '.format(
                  result.success, result.message))

        #% Obtain Theta1 and Theta2 back from nn_params
        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                   (hidden_layer_size, (input_layer_size + 1)), 
                    order = 'F')

        Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                   (num_labels, (hidden_layer_size + 1)), 
                   order = 'F')  


        # Pause
        #print("Program paused. Press Ctrl-D to continue.\n")
        #code.interact(local=dict(globals(), **locals()))
        #print(" ... continuing\n ")

        # jkmm - pauses in here
        # visualizeNN(Theta1, hidden_layer_size, MAXITER, _lambda)


        #%% ================= Part 3 Predict =================
        #   it to get a good _lambda and other parms
        pred = predict(Theta1, Theta2, X_cv)

        # display a sample of the data & corresponding predicted labels
        # displaySampleData(X_cv, np.vstack(pred))  # jkmm - comment out  

        ''' above displaySample does this so comment out for now
        # print out what the images are predicted to be
        print('Selecting random examples of the data to display and how they are predicted.\n')
        print(pred[sel].reshape(10, 10))

        # display the sample images
        displayData(images)
        '''

        # JKM - my array was column stacked - don't understand why this works
        pp = np.row_stack(pred)
        accuracy = np.mean(np.double(pp == y_cv)) * 100

        print('\n Cross Valid Set Accuracy: {0} \n'.format(accuracy))
        print('\n  Parms: Hidden Layer Units: {0}  Max Iters: {1}  Lambda: {2}  \n'.format( 
                    hidden_layer_size, MAXITER, _lambda))


        # Create a filname to use to write the results
        fn = 'HU_{0}_MaxIter_{1}_Lambda_{2}_PredAcc_{3}'.format(
               hidden_layer_size, MAXITER, _lambda, accuracy)

        # Capture the Thetas
        writeLearnedTheta(Theta1, 'Theta1_' + fn)
        writeLearnedTheta(Theta2, 'Theta2_' + fn)
 
        if (accuracy > best_acc):
            best_acc = accuracy
            best_lambda = _lambda
            print 'updating _lambda & best_acc'
            
    # end of _lambda loop
    print 'end of _lambda loop'

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    '''



    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")     

#%% ================= Part 12: Run predictionon Test data  =================
#      Run teh prediction on teh test data 
#      write the results to a csv file to submit to kaggle 
#

    ## jkmm - Get resulst with X_test, Y_test

    # jkm - move this part to a separate script that reads the theta's
    #   and calculates teh results. 

    ## NEXT PART
    kagglePred = obtainKaggleTestResults(Theta1, Theta2)    
    writeKaggleTestResults(fn, kagglePred)

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    return

'''

# ========================================    
# Go to main if calling from command line
if __name__ == "__main__":
    print("Going to main")
    main()
