import sys
import code

import pandas as pd
import numpy as np

# Locals
from Stage1.pred import predict
from Stage1.displayData import displayData

def obtainKaggleTestResults(Theta1, Theta2):
    ''' Apply the learned neural network paramaters to predict the
        results on the Kaggle Test Data.
    '''

    # get the data to do the final test.
    m, n, X = getKaggleTestData()

    # Select some random images from X
    print('Selecting random examples of the test data to display.\n')
    sel = np.random.permutation(m)
    sel = sel[0:100]
    images = X[sel, :]

    # display the sample images
    displayData(images)

    # predict the labels
    pred = predict(Theta1, Theta2, X)

    # print out what the images are predicted to be
    print('Here are the predicted labels for these random images: \n')
    print(pred[sel].reshape(10, 10))

    # Pause
    print("Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    return pred


def writeKaggleTestResults(fn, pred):
    ''' Write the predicted label results in the Kaggle format'''

    resultsDir = '/home/jennym/Kaggle/DigitRecognizer/Data/Results/'
    fnExt = '.csv' 
 
    #Create the csv file name
    fnResults = resultsDir + 'ResultsFor_' + fn + fnExt 

    # Create the dataframe to write
    df = pd.DataFrame(np.vstack(pred)) # 2D array e.g. column stacked 
    df.to_csv(path_or_buf=fnResults, header=['Label'], index=True, index_label='ImageId')

    print('\n Results written to File:\n  {0}\n'.format(fnResults))

    return


def getKaggleTestData():
    ''' Get the test data based on the Kaggle format 
        Returns:
           m: Number of rows
           n: number of features
           X: training examples
     ''' 
     # TODO - make this and reading the files more generic.
     #         can return an empty Y 
 

    fnTest = '/home/jennym/Kaggle/DigitRecognizer/ex4/test.csv'

    # Get the Data to test on.
    print("\n Loading in the Kaggle Test Data:\n ... " + fnTest + " ...\n")
    test = pd.read_csv(fnTest)

    M = test.as_matrix() # gives numpy array
    X = M[:,0:]    # The test data

    # Scale the input grey scale pixels by 255 to between 0 and 1 float
    X = X.astype('float')
    X = X/X.max()
 
    # Find number of rows examples "m", & columns features "n" 
    m, n = X.shape 

    return m, n, X   

'''
# old matlab code below
% There is no label y,so comment out this part. 
%  y = X(:, 1);
%  Replace the 0's with 10 in y for indexing purposes
%  y(y==0) = 10;
% X = X(:, 2:end);

m = size(TEST, 1);

% Re-orient the images
for i = 1: m 
    TEST(i, :) = reshape(reshape(TEST(i, :), ... 
        image_size, image_size)', ...
        1, input_layer_size);
end    

% Scale the data to be floating point numbers from 0-1
TEST = double(TEST) / 255 ;

% Randomly select 100 data points to display
sel = randperm(size(TEST, 1));
sel = sel(1:100);
displayData(TEST(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\n Predicting results for Kaggle Test Data ... \n');
predTest = predict(Theta1, Theta2, TEST);

% Replace the 10's with 0 in the prediction results.
predTest(predTest==10) = 0;

% Print the predicted values of the displayed data
fprintf('\nSample of predictions of TEST data:\n')
b = predTest(sel, :);
reshape(b, 10, 10)'

%Create the csv file name
fn = sprintf('HiddenUnits_%d_MaxIter_%d_Lambda_%d_TrainAcc_%f.csv', ...
    hidden_layer_size, maxIterTrain, lambdaTrain, accuracyTrain);

% Write to a csv file.
csvwrite(fn, predTest);

fprintf('\nResults written to Kaggle File: %s\n', fn)
'''
