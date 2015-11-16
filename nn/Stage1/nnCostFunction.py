import numpy as np
import code  # temp, only needed to pause here fore debugging


# Locals Import
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, MLlambda):
    ''' Some comments.
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
% [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, MLlambda) 
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
'''
    # make sure all further math with MLlambda is done as float,
    #  sometimes caller sets MLlambda to be an int
    MLlambda = float(MLlambda)  

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
               (hidden_layer_size, (input_layer_size + 1)), 
                order = 'F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
               (num_labels, (hidden_layer_size + 1)), 
               order = 'F')  

    # % Setup some useful variables, num examples and features
    m, n = np.shape(X) 
         
    #% You need to return the following variables correctly 
    J = 0;
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

    #% Compute Cost of feed forward.

    # create a 10x10 eye matrix of ones.
    y_eye = np.eye(num_labels)
    y_matrix = y_eye[y[:,0], :] # y_matrix = y_eye(y, :); 

    # % Calculate cost. Assuming a 3 layer neural network. 
    # % Add ones column to the X data matrix
    a1 = np.c_[np.ones(m), X]

    #% Calculate a2 outputs for hidden layer
    z2 = np.dot(a1 , Theta1.transpose())   # m X 25
    a2 = sigmoid(z2)    # m x 25
    a2 = np.c_[np.ones(m), a2] # add a0 = 1, column of 1's -? m x 26

    z3 = np.dot(a2, Theta2.transpose())  # m x 10
    a3 = sigmoid(z3)  # m x 10

    hox = a3

    Inner_J = -y_matrix*np.log(hox) - (1 - y_matrix)*np.log(1 - hox)

    J_wo_reg = np.sum(Inner_J)/m #J_wo_reg = sum(sum(Inner_J))/m;


    #% Calculate Regularization portion
    Theta1_no_bias = Theta1[:, 1:]
    Theta2_no_bias = Theta2[:, 1:]

    Theta1_no_bias_squared = np.square(Theta1_no_bias)
    Theta2_no_bias_squared = np.square(Theta2_no_bias)

    reg = (float(MLlambda)/(2*m)) * ( sum(sum(Theta1_no_bias_squared)) + 
                                      sum(sum(Theta2_no_bias_squared)))
    J = J_wo_reg + reg 


#% ***************************************************************
#% ************************* PART 2 ******************************
#% ***************************************************************

    #% Calculate the gradients
    #% Assuming a 3 layer network.

    #% STEP 1: Calculate error at level 3: d3
    d3 = a3 - y_matrix

    #% STEP 2: Calculate error at Level 2: d2
    siggrad_z2 = sigmoidGradient(z2) 
    # % NOTE: a'b = ba
    # d2 = (d3 * Theta2_no_bias).*siggrad_z2;
    d2 = np.dot(d3,Theta2_no_bias) * siggrad_z2

    #% STEP 3: Calculate Delta's:  Delta1 & Delta2 (ie the triangles)
    #% Note, have already removed bias unit in Delta1 prior, as 
    #%  d2 was computed with Theta2 with bias removed.
    Delta1 = np.dot(d2.transpose(), a1)
    Delta2 = np.dot(d3.transpose(), a2)  

    #% Calculate the back prop gradients.
    Theta1_grad = (1./m)* Delta1
    Theta2_grad = (1./m)* Delta2 

    ''' 
% ***************************************************************
% ************************* PART 3 ******************************
% ***************************************************************

% Calculate regularization component of the gradient.
%  Theta1 and Theta2 include the bias components, but to 
%  calculate the regularization, we do not want to include
%  the bias. So we zero out the bias columns, so it will have
%  no impact when we add it to the gradient that was calculated 
%  above (e.g. without regularization). But we want to keep the
%  matrix sizes the same so we can do the additions using vector
%  or matrix math.
'''

    #% Zero out the bias unit in Theta1
    Theta1_bias_zero = np.copy(Theta1)
    Theta1_bias_zero[:, 0] = 0

    #% Zero out the bias unit in Theta2
    Theta2_bias_zero = np.copy(Theta2)
    Theta2_bias_zero[:, 0] = 0

    #% Scale Theta's by lambda/m 
    Theta1_reg = (MLlambda/m ) * Theta1_bias_zero
    Theta2_reg = (MLlambda/m ) * Theta2_bias_zero

    #% Add regularization component to the gradients
    Theta1_grad = Theta1_grad + Theta1_reg
    Theta2_grad = Theta2_grad + Theta2_reg

    #% Unroll gradients
    #grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad = np.hstack((Theta1_grad.ravel(order='F'), Theta2_grad.ravel(order='F')))


    # JKMM pause for debug
    #print("JKMM Program paused in nnCostFunction. Press Ctrl-D to continue.\n")
    #code.interact(local=dict(globals(), **locals()))
    #print(" ... continuing\n ")  

    return J, grad
