## THIS FILE IS NOT USED. 
##  Functions are in file checkNNgradients.py

import numpy as np

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
