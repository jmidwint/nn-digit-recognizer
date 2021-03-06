#!/usr/bin/python

import sys
import code

import math
import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import scipy.optimize as sci

# Local files
from displayData import displayData
from getMatlabTrainingData import getMatlabTrainingData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from pred import predict


# Global Variables

# input_layer_size = 
# NN Architecture Parms 
hidden_layer_size = 25
num_labels = 10 
MAXITER = 400 # Iterations for learning

# ===================================================
#
#                   ===   MAIN   ===
# 
#====================================================

def main():
    ''' Main function  '''

    ## %% =========== Part 1: Loading and Visualizing Data =============
    #%  We start the exercise by first loading and visualizing the dataset. 
    #%  You will be working with a dataset that contains handwritten digits.
    #%


    # Read the Matlab data
    m, n, X, y = getMatlabTrainingData()

    # number of features
    input_layer_size = n    
 

    # Select some random images from X
    print('Selecting random examples of the data to display.\n')
    sel = np.random.permutation(m)
    sel = sel[0:100]
    
    #  Re-work the data orientation of each training example
    image_size = 20
    XMatlab = np.copy(X) # Need a deep copy, not just the reference
    for i in range(m): 
        XMatlab[i, :] = XMatlab[i, :].reshape(image_size, image_size).transpose().reshape(1, image_size*image_size)

    # display the sample images
    displayData(XMatlab[sel, :])

    # Print Out the labels for what is being seen. 
    print('These are the labels for the data ...\n')
    print(y[sel, :].reshape(10, 10))

    # Pause program
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  


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
    MLlambda = 0.0

    # Cluge, put y back to matlab version, then adjust to use python
    #  indexing later into y_matrix
    y[(y == 0)] = 10
    y = y - 1
    J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, MLlambda)

    print('Cost at parameters (loaded from ex4weights): ' + str(J) + 
          '\n (this value should be about 0.287629)\n')

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

#%% =============== Part 4: Implement Regularization ===============
#%  Once your cost function implementation is correct, you should now
#%  continue to implement the regularization with the cost.
#%

    print('\nChecking Cost Function (with Regularization) ... \n')

    # % Weight regularization parameter (we set this to 1 here).
    MLlambda = 1.0

    J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, MLlambda)

    print('Cost at parameters (loaded from ex4weights): ' + str(J) +
         '\n(this value should be about 0.383770)\n');

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  


#%% ================ Part 5: Sigmoid Gradient  ================
#%  Before you start implementing the neural network, you will first
#%  implement the gradient for the sigmoid function. You should complete the
#%  code in the sigmoidGradient.m file.
#%

    print('\nEvaluating sigmoid gradient...\n')
    g = sigmoidGradient(np.array([1, -0.5,  0,  0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')
    print(g)
    print('\n\n')

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

 
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


#%% =============== Part 7: Implement Backpropagation ===============
#%  Once your cost matches up with ours, you should proceed to implement the
#%  backpropagation algorithm for the neural network. You should add to the
#%  code you've written in nnCostFunction.m to return the partial
#%  derivatives of the parameters.
#%
    print('\nChecking Backpropagation... \n')

    #%  Check gradients by running checkNNGradients
    checkNNGradients()

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

#%% =============== Part 8: Implement Regularization ===============
#%  Once your backpropagation implementation is correct, you should now
#%  continue to implement the regularization with the cost and gradient.
#%

    print('\nChecking Backpropagation (w/ Regularization) ... \n')

    #%  Check gradients by running checkNNGradients
    MLlambda = 3
    checkNNGradients(MLlambda)

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  

    #% Also output the costFunction debugging values
    debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, MLlambda)

    print('\n\n Cost at (fixed) debugging parameters (w/ lambda = ' + 
          '{0}): {1}'.format(MLlambda, debug_J))
    print('\n  (this value should be about 0.576051)\n\n')

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

#%% =================== Part 8b: Training NN ===================
#%  You have now implemented all the code necessary to train a neural 
#%  network. To train your neural network, we will now use "fmincg", which
#%  is a function which works similarly to "fminunc". Recall that these
#%  advanced optimizers are able to train our cost functions efficiently as
#%  long as we provide them with the gradient computations.
#%
    print ('\nTraining Neural Network... \n')

    #%  After you have completed the assignment, change the MaxIter to a larger
    #%  value to see how more training helps.
    #% jkm change maxIter from 50-> 400
    options = {'maxiter': MAXITER}

    #%  You should also try different values of lambda
    MLlambda = 1

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

    print('\nVisualizing Neural Network... \n')

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

    # JKM - my array was column stacked - don't understand why this works
    pp = np.row_stack(pred)
    accuracy = np.mean(np.double(pp == y)) * 100

    print('\nTraining Set Accuracy: {0} \n'.format(accuracy))

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

  
# ========================================

    # All Done!
    return

# ========================================   
# Go to main if calling from command line
if __name__ == "__main__":
    print("Going to main")
    main()
