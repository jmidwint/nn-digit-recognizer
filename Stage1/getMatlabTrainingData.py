import scipy.io as sio

def getMatlabTrainingData():
    ''' Get the data based on the Coursera format
        Returns:
           m: Number of rows
           n: number of features
           X: training examples
           y: labels for X ''' 

    fnTrain = '/home/jennym/Kaggle/DigitRecognizer/ex4/ex4data1.mat'
    print("\n Reading in the training file:\n ... " + fnTrain + " ...\n")
    train = sio.loadmat(fnTrain)
    
    X = train['X']
    y = train['y']
    (m, n) = X.shape


    # Replace the 10's with 0 in labels y for indexing purposes
    y[(y == 10)] = 0

    return m, n, X, y

