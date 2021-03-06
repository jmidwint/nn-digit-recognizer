
import pandas as pd
import numpy as np

def writeLearnedTheta(Theta, fn):
    ''' Write the learned Theta value to a file '''

    thetasDir = '/home/jennym/Kaggle/DigitRecognizer/Data/LearnedThetas/'
    fnExt = '.csv'
    fnTheta = thetasDir + fn + fnExt 
 
    # Create the dataframe to write
    df = pd.DataFrame(Theta) # 2D array e.g. column stacked 
    df.to_csv(path_or_buf = fnTheta ) # Row and column are numbered

    print('\n Learned Theta written to File:\n  {0}\n'.format(fnTheta))

    return

    # TODO - re-write this as a class for hndling actions related to theta's
