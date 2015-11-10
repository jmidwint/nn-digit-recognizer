import numpy as np

# Local imports
from nnCostFunction import nnCostFunction 

def checkNNGradients(MLlambda = 0):
    '''
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%
'''
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    #% We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    #% Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    #y  = 1 + mod(1:m, num_labels)'
    y  = 1 + (np.arange(m)+1) % num_labels

    # JKM - need a column vector
    y = y[:, None]
    y = y - 1

    #% Unroll parameters
    nn_params = np.hstack((Theta1.ravel(order = 'F'), Theta2.ravel(order = 'F')))

    #% Short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                               num_labels, X, y, MLlambda)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    #% Visually examine the two gradient computations.  The two columns
    #% you get should be very similar.
    print np.column_stack((numgrad, grad)) 
    # disp([numgrad grad]) -> matlab
    print('The above two columns you get should be very similar.\n' + 
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    #% Evaluate the norm of the difference between two solutions.  
    #% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    #% in computeNumericalGradient.m, then diff below should be less than 1e-9
    # diff = norm(numgrad-grad)/norm(numgrad+grad)
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)

   

    print(" If your backpropagation implementation is correct, then")
    print(" the relative difference will be small (less than 1e-9).")
    print "  Relative Difference: ", diff
    print("\n\n")

    return

#=================================================================
def debugInitializeWeights(fan_out, fan_in):
    '''
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%
'''
    #% Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    #% Initialize W using "sin", this ensures that W is always of the same
    #% values and will be useful for debugging
    # W = np.reshape(sin(1:numel(W)), size(W)) / 10;
    W = np.reshape(np.sin(np.arange(W.size) +1 ), np.shape(W), order='F') / 10
    return W

#=================================================================

def computeNumericalGradient(J, theta):
    '''
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
%                
'''

    numgrad = np.zeros(np.shape(theta))
    perturb = np.zeros(np.shape(theta))
    e = 1e-4
    for p in np.ndindex(np.shape(theta)):
        #% Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        #% Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad
