import pandas as pd
import numpy as np

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
