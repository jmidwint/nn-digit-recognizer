The code here is exploring the hyper-parm space of convolutional
networks.


conv1.py:
  - This is a straight copy of the original convolutional.py code from 
     Tensorflow.


conv2.py:
  - Based on original TensorFlow convolutional.py code, but modified to
    work with Kaggle data, and with my previous code for looking at 
    datasets images, and to capture test results for Kaggle submission.
  - Gets a .98986 accuracy on Kaggle.   


conv3.py:
  - Based on original TensorFlow convolutional.py code, but modified to
  expand the MNIST dataset. (cloned from conv1.py in this directory).
  - Expansion is done on the MNIST 60,000 training examples, by moving 
  the pixels in each image up/down/left/right by one pixels, to create 
  a new training set of 300,000 examples. 
  - the # of epochs was reduced to 5 (from 10) with result: 0.4 test error 
     on one run (takes about 5 hours on my machine)
  - the # epochs returned to orginal 10 with result: 0.5 test error 
     on one run 


conv4.py:
  - cloned from conv1.py to work with Kaggle training data,and test with MNIST TEST data,
     & prediction done on Kaggle Test Data
     Results on 10 Epochs : Test error 0.4 , on 1/1 run (about 1 hour)
  - modified to expand the training set data, as per conv3.py, where 
     the pixels in each image up/down/left/right by one pixels, to create 
     a new training set of 210,000 examples.
     Results on 10 Epochs : Test error 0.2 (.18), on 1/1 run (took 6 hours)
     Kaggle Test Result: 0.99300
   ** This version has a bug, discovered later, in which last 32 samples 
      of the prediction with Kaggle Test Data are left out. This would 
      of course affect the Test Results. This is fixed on conv4v2.py

conv4v2.py:
  - copied from conv4.py, and fixed so as to be able to generate test results
    with MNIST data via "batches". This was done so as to avoid getting 
    an OOM (Out of memory) error when trying to run this code on AWS T2
    instance (Amazon web service "free tier" for cloud computing). The 
    tensor matrices/vectors being loaded in to process, were too large. 
    Note: The T2 instance was only being used to explore being able to run 
    on AWS, as a pre-cursor to eventually trying to run on faster GPU's.     
  - also fixes a "day 1" bug in my code with batch processing where if
    the number of samples per batch did not go evenly into the total number
    of samples, then the samples in the very last set would not be processed.
    This would affect test results and skew the error rate actually making
    it look worse then it should have been.  

conv4v3.py:
   - copied from conv4v2.py
   - enhanced for determining the best number of epochs
   - captures interim results 
    
   
  
