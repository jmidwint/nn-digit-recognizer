
import numpy as np

# Local imports
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    ''' PREDICT Predict the label of an input given a trained neural network
    %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    %   trained weights of a neural network (Theta1, Theta2)
    '''

    #% Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # % You need to return the following variables correctly 
    p = np.zeros(m)

 
    h1 = sigmoid(np.dot(np.c_[np.ones(m), X], Theta1.transpose()))
    h2 = sigmoid(np.dot(np.c_[np.ones(m), h1], Theta2.transpose()))
    p = np.argmax(h2, axis = 1) # + 1 JKMM - should I add this and then add 1+y later?

    return p
