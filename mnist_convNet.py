import tensorflow as tf
import numpy as np
import pylab
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

"""
    Three steps to use tensorflow.
    1 - Build the inference Graph
    2 - Build the loss graph and return the loss tensor
    3 - Write a training function that apply the gradient updates
"""

def initializeWeightsConvLayer(shape):
    """
        Initialize the weights of a convolutional layer
        args :
            its shape
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01), tf.float32)

def initializeBiasesConvLayer(shape):
    """
        Initialize the biases of a convolutional layer
        args :
        its shape
    """
    return tf.Variable(tf.zeros(shape), tf.float32)

def convNetModel(images, weightsShape, biasesShape, numberClasses):
    """
        Model of a CNN
        args :
            images : training data
            weightsShape : dictionary with the shape of the weights each convolutional layers
            biasesShape : dictionary with the shape of the biases each convolutional layers
            numberClasses : number of classes to be predicted
        return
            logits : scores of predicted labels
    """
    # We use a conv net hence the input has to be a 2D image : reshape
    images = tf.reshape(images, [-1, 28, 28,1])
    
    with tf.name_scope("ConvLayer1"):
        weights = initializeWeightsConvLayer(weightsShape['conv1'])
        biases = initializeBiasesConvLayer(biasesShape['conv1'])
        
        #Conv layer 1 (?, 28, 28, 32)
        #Apply a padding of 1 pixel in each dimension
        convLayer1 = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding="SAME")
        biasConvLayer1 = tf.nn.bias_add(convLayer1, biases)

        #Apply RELU and then pool
        #Pool layer 1 (?, 14, 14, 32)
        reluLayer1 = tf.nn.relu(biasConvLayer1)
        poolLayer1 = tf.nn.max_pool(reluLayer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    with tf.name_scope("ConvLayer2"):
        weights = initializeWeightsConvLayer(weightsShape['conv2'])
        biases = initializeBiasesConvLayer(biasesShape['conv2'])
        
        #Conv layer 1 (?, 28, 28, 32)
        #Apply a padding of 1 pixel in each dimension
        convLayer2= tf.nn.conv2d(poolLayer1, weights, strides=[1,1,1,1], padding="SAME")
        biasConvLayer2 = tf.nn.bias_add(convLayer2, biases)
        
        #Apply RELU and then pool
        #Pool layer 2 (?, 7, 7, 32)
        reluLayer2 = tf.nn.relu(biasConvLayer2)
        poolLayer2 = tf.nn.max_pool(reluLayer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.name_scope("ConvLayer3"):
        weights = initializeWeightsConvLayer(weightsShape['conv3'])
        biases = initializeBiasesConvLayer(biasesShape['conv3'])
        
        #Conv layer 2 (?, 14, 14, 64)
        #Apply a padding of 1 pixel in each dimension
        convLayer3= tf.nn.conv2d(poolLayer2, weights, strides=[1,1,1,1], padding="SAME")
        biasConvLayer3 = tf.nn.bias_add(convLayer3, biases)
        
        #Apply RELU and then pool
        #Pool layer 3 (?, 4, 4, 128)
        reluLayer3 = tf.nn.relu(biasConvLayer3)
        poolLayer3 = tf.nn.max_pool(reluLayer3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.name_scope("FullyConnected"):
        poolLayer3 = tf.reshape(poolLayer3, [-1, weightsShape['FC1'][0]]) #Reshape into (?, 4*4*128)

        weights = tf.Variable(tf.random_normal(weightsShape['FC1'], stddev=0.01), name="weights")
        biases = tf.Variable(tf.zeros(biasesShape['FC1']), name="biases")
        
        #Fully connected layer
        fullyConnectedLayer = tf.nn.relu(tf.matmul(poolLayer3, weights)+biases)

    with tf.name_scope("SoftmaxLayer"):
        weights = tf.Variable(tf.random_normal(weightsShape['softmax'], stddev=0.01), name="weights")
        biases = tf.Variable(tf.zeros([numberClasses]), name="biases")
        
        #Softmax
        logits = tf.matmul(fullyConnectedLayer, weights)+biases

    return logits

def lossFunction(labels, logits):
    """
        Loss function
        args :
        labels : training data labels
        logits : score of predicted data labels
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def training(learningRate, images, labels, weightsShape, biasesShape, numberClasses):
    """
        Trains the CNN with gradient descent
        args :
            learningRate : learning rate
            images : input training data
            labels : input training data labels
            weightsShape : dictionary with the shape of the weights each convolutional layers
            biasesShape : dictionary with the shape of the biases each convolutional layers
            numberClasses : number of classes to be predicted
    """
    logits = convNetModel(images, weightsShape, biasesShape, numberClasses)
    loss = lossFunction(labels, logits)
    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    return train_step, loss, logits

def main():
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    DIMENSION = IMAGE_WIDTH*IMAGE_HEIGHT
    NUMBER_CLASSES = 10
    
    #load mnist and reshape
    TRAIN_DIR = "MNIST_data"
    mnist = read_data_sets(TRAIN_DIR, one_hot = True)
 
    #Architecture (CONV2D->RELU->POOL)^3->FC->softmax
    weightsShape = {
        #32 3x3 filter on a grayscale image 28x28x1
        'conv1' : [3,3,1,32],
        #image 14x14x32
        'conv2' :[3, 3, 32, 64],
        'conv3' :[3, 3, 64, 128],
        'FC1' : [4*4*128, 625],
        'softmax' : [625, 10],
    }

    biasesShape = {
        #32 3x3 filter on a grayscale image 28x28x1
        'conv1' : [32],
        #image 14x14x32
        'conv2' :[64],
        'conv3' :[128],
        'FC1' : [625],
        'softmax' : [10],
    }

    MAX_TRAINING_STEP = 3000
    BATCH_SIZE = 100
    
    #Placeholders for data
    xData = tf.placeholder(tf.float32, [None, DIMENSION])
    yData = tf.placeholder(tf.float32, [None, NUMBER_CLASSES])
    
    #Compute the model with the placeholders
    train_step, loss, logits = training(0.5, xData, yData, weightsShape, biasesShape, NUMBER_CLASSES)
    
    
    #Tensorflow session
    #Initialize variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    #Start training on the model
    lossValues = []
    for i in range(MAX_TRAINING_STEP):
        xBatch, yBatch  = mnist.train.next_batch(BATCH_SIZE)
        _, currentLoss = sess.run([train_step, loss], {xData:xBatch, yData:yBatch})
        if i%100 == 0:
            print('%d loss %f' % (i, currentLoss))
        lossValues.append(currentLoss)
    
    yPrediction = tf.nn.softmax(logits)
    correctPrediction = tf.equal(tf.argmax(yData, axis=1), tf.argmax(yPrediction, axis=1)) 
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
	
    #Evaluate the accuracy by batches to avoid memory problems
    finalAccuracy = 0.0
    for i in range(100):
        xBatch, yBatch  = mnist.test.next_batch(BATCH_SIZE)
        finalAccuracy += BATCH_SIZE*sess.run(accuracy, {xData: xBatch, yData: yBatch})
		
    finalAccuracy = finalAccuracy/10000.0
    print("Accuracy %f ", finalAccuracy)

    plt.plot(lossValues)
    plt.show()
	
if __name__ == "__main__":
    main()

