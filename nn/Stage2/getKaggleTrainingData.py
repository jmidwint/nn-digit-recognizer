import pandas as pd
import numpy as np

from Stage1.displayData import displayData

def getKaggleTrainingData(scale = 1.0, stats = True):
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
    if stats:
        print(" Some raw stats on the data here: \n")
        print(train.describe())

    M = train.as_matrix() # gives numpy array

    X = M[:,1:]    # The training data
    y = M[:, 0:1]  # The labels or answers

    # Scale the input grey scale pixels to be between -1  and 1 float
    X = X.astype('float')
    X = (X - ( X.max() /float(scale)))/X.max()
 
    # Find number of rows examples "m", & columns features "n" 
    m, n = X.shape 

    return m, n, X, y    


def displaySampleData(X, y):
    ''' Display random images as a grid, along with the corresponding labels'''
    m, n = X.shape
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


# For partitioning the data    
TRAIN_PERCENT = 70
CV_PERCENT = 15
TEST_PERCENT = 15
 
def partitionData(X, y):
    ''' some comments'''
    #
    m = X.shape[0]
    m_train = m * TRAIN_PERCENT/100
    m_cv = m * CV_PERCENT/100
    #
    # Create index of random indices   
    idx = np.random.permutation(m)
    idx_train = idx[0: m_train] 
    idx_cv = idx[m_train: (m_train + m_cv)]
    idx_test = idx[(m_train + m_cv) : ]
    #
    return X[idx_train, :], y[idx_train], X[idx_cv, :],  y[idx_cv], X[idx_test, :], y[idx_test] 


def countData(X):
    ''' manually counts the disribution of the data ''' 
    for i in np.arange(10):
        print('{0}: {1}'.format(i, (sum(X == i))))
    
