"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.

JKM - copied from the google TensorFlow and modified for my purpose. 

"""
import gzip
import os
import sys
import urllib

import tensorflow.python.platform

import numpy
import tensorflow as tf

# jkm imports
import code
from Stage2.getKaggleTrainingData import getKaggleTrainingData, displaySampleData, partitionData
from Stage2.obtainKaggleTestResults import getKaggleTestData, writeKaggleTestResults
from Stage1.displayData import displayData

# jkm - don't need this since I already have the data downloaded ina different form elswhere
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/home/jennym/data' # jkmm
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set. jkm - look at this ** 3000 is better 
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 100 # jkm - changed from 10

# JKM
KAGGLE_BATCH_SIZE = 100

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print 'Succesfully downloaded', filename, statinfo.st_size, 'bytes.'
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

# JKM - added this
def convert_data(X, num_images):
    ''' Convert the images into a 4D tensor [image index, y, x, channels].
     where y and x are 28x28 pixel images
    '''
    print 'Converting Data'
    X = X.astype(numpy.float32)
    X = X.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return X

# JKM - added this
def convert_labels(labels):
    ''' Convert the lables into a 1-hot matrix [image index, label index].'''
    print 'Converting labels'
    return (numpy.arange(NUM_LABELS) == labels[:]).astype(numpy.float32)

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = numpy.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images, NUM_LABELS), dtype=numpy.float32)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image, label] = 1.0
    return data, labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.self_test:
        print 'Running self-test.'
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(16)
        test_data, test_labels = fake_data(256)
        num_epochs = 1
    else:
        # Get the data. #### MNIST DATA ####
        ''' JKM - added tf_
        tf_train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
        tf_train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        tf_test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
        tf_test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
 
        # Extract it into numpy arrays.
        # jkm - rename to tf_  
        tf_train_data = extract_data(tf_train_data_filename, 60000)
        tf_train_labels = extract_labels(tf_train_labels_filename, 60000)
        tf_test_data = extract_data(tf_test_data_filename, 10000)
        tf_test_labels = extract_labels(tf_test_labels_filename, 10000)
        
        # Generate a validation set JKM - adding tf_
        tf_validation_data = tf_train_data[:VALIDATION_SIZE, :, :, :]
        tf_validation_labels = tf_train_labels[:VALIDATION_SIZE]
        tf_train_data = tf_train_data[VALIDATION_SIZE:, :, :, :]
        tf_train_labels = tf_train_labels[VALIDATION_SIZE:]
        '''

        ### KAGGLE DATA #####
        ## JKM - get Kaggle comp data & convert to 4D tensor data
        m, n, X, y = getKaggleTrainingData(scale = 2.0, stats = False)

        # JKM - display a sample of the data & corresponding labels
        #displaySampleData(X, y)

        # JKM- partition my X data set into 3 sets, X, CV, TEST
        # Partition the Kaggle training data, randomly
        X_train, y_train, X_cv, y_cv, X_test, y_test =  partitionData(X, y)   

        # JKM-Convert to 4D tensor data and 1-hot matrix labels
        train_data = convert_data(X_train, X_train.shape[0])
        train_labels = convert_labels(y_train)
        validation_data = convert_data(X_cv, X_cv.shape[0])
        validation_labels = convert_labels(y_cv)
        test_data = convert_data(X_test, X_test.shape[0])
        test_labels = convert_labels(y_test)  
 
        # JKM- Get the Kaggle test data
        # get the data to do the final test.
        m_KaggleTest, n_KaggleTest, X_KaggleTest = getKaggleTestData(scale = 2)
        kaggle_test_data = convert_data(X_KaggleTest, m_KaggleTest)
 
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    # jkm - create a node for the kaggle data
    # create 2 types of nodes, one regular and one batch
    kaggle_test_data_node = tf.constant(kaggle_test_data)
    batch_kaggle_test_data_node = tf.placeholder(
                            tf.float32,
                            shape=(KAGGLE_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))

    # jkm - set the prediction. debug - I have 2, one regular and one batch
    kaggle_test_prediction = tf.nn.softmax(model(kaggle_test_data_node))
    batch_kaggle_test_prediction = tf.nn.softmax(model(batch_kaggle_test_data_node))
    
    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels)
                print 'Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(), validation_labels)
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(test_prediction.eval(), test_labels)
        print 'Test error: %.1f%%' % test_error
        if FLAGS.self_test:
            print 'test_error', test_error
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                test_error,)
        # jkm - Get the Kaggle test results, need to use the kaggle batch node/model
        #kaggle_pred = kaggle_test_prediction.eval()
        #KAGGLE_BATCH_SIZE = 100
        kaggle_result = numpy.zeros(m_KaggleTest)
        batch_test_size = m_KaggleTest # jkm - don't forget the last less then 64
        for step in xrange(batch_test_size / KAGGLE_BATCH_SIZE):
            offset = step * KAGGLE_BATCH_SIZE
            k_batch_data = kaggle_test_data[offset:(offset + KAGGLE_BATCH_SIZE), :, :, :]
            feed_dict = { batch_kaggle_test_data_node: k_batch_data }
            (batch_predictions,) = s.run([batch_kaggle_test_prediction], feed_dict=feed_dict)
            kaggle_result[offset:(offset + KAGGLE_BATCH_SIZE)] = numpy.argmax(batch_predictions, 1)        
        # jkm - Display some of the results
        displaySampleData(X_KaggleTest, numpy.vstack(kaggle_result))
        # jkm - write the results
        # Create a filname to use to write the results
        fn = 'CNN_Num_Epochs_{0}_Last_LR_{1}_Test_Error_{2}'.format(
                num_epochs, lr, test_error)
        writeKaggleTestResults(fn, kaggle_result)
        # Pause - jkm
        print("9__Program paused. Press Ctrl-D to continue.\n")
        code.interact(local=dict(globals(), **locals()))
        print(" ... continuing\n ")  


if __name__ == '__main__':
    tf.app.run()
