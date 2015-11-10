# nn-digit-recognizer
Exploring neural network implementations for handwritten digit recognition (classic MNIST problem).

This repository is a placeholder for my attempts to implement various artificial neural network solutions to the classic handwritten digit recognizer problem. The primary purpose is two fold: 

  1) Develop an understanding of artifical neural networks as applied to solve  supervised machine learning applications for image recognition. 
     
  2) Develop an understanding of the techniques for python implementation, including use of various pre-existing python libraries, or related other open source platforms that have been created to support artifical neural network implementations. 
     
The development is planned in stages  to facilitate both learning and to capture working incremental solutions.

Stage 1: Port MATLAB Solution to Python - for ML Matlab Course Assignment Based Digit Images
  
  The first stage is to port or convert my original solution, that was done as part of a separate Machine Learning (ML) course,  from MATLAB to python. This implementation is a simple 3 layer artificial neural network that processes labelled digit images that were originally taken from the MNIST database and that were subsequently modified for the MATLAB ML course assignment. These images are 20 x 20 pixelated labelled 0-9 grey scaled images that have already been feature scaled be between -1 to 1. 
    
Stage 2:  Add Support for Kaggle Competition Based Images
  
  The second stage enhances the solution to process a set of training and test images that were provided as part of the Digit Recognizer competion on Kaggle.com. Images are 28 x 28 grey-scaled images that range from 0-255 (not feature scaled). This solution is used to create a submission file for the kaggle competition.
  
Stage 3: Investigate Implementation of Convolutional Neural Network Solution
  
  The idea here is to develop a solution to this simple Digit recognizer problem using artificial "convolutional" neural network, which is a more advanced state-of-the-art neural network solution being used in image recognition, that to date has yielded  higher prediction rate for this problem space. This stage will also investigate the use of open source platforms/tools/libraries that have been developed to support the implementation of artifical neural networks. As well this stage should also explore the use of Graphical Processing Units (GPU's) to reduce the processing time required for the training of a neural network to recognize images.   
  
Updates and  further details on this staged plan to be provided as the development proceeds and more is being understood. 

