import os
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

def mnist_inferenceGraph(images, dimension, hiddenUnits1, hiddenUnits2, numberClasses):
    """Build the MNIST inference graph
    args:
        images : images placeholder (data)
        dimension : size of the images
        hiddenUnits1 : Size of the first layer of hidden units
        hiddenUnits2 : Size of the second layer of hidden units
        numberClasses : number of classes to classify
    outputs:
        tensor with the computed logits
    """
        
    with tf.name_scope("hiddenLayer1"):
        weights = tf.Variable(tf.truncated_normal([dimension, hiddenUnits1], stddev = 1.0/tf.sqrt(float(dimension)),
                                      name ="weights"))
        biases = tf.Variable(tf.zeros([hiddenUnits1], name="biases"))
        
        activationHidden1 = tf.nn.relu(tf.matmul(images, weights)+biases)

    with tf.name_scope("hiddenLayer2"):
        weights = tf.Variable(tf.truncated_normal([hiddenUnits1, hiddenUnits2], stddev = 1.0/tf.sqrt(float(hiddenUnits1)),
                              name="weights"))
        biases = tf.Variable(tf.zeros([hiddenUnits2]), name="biases")

        activationHidden2 = tf.nn.relu(tf.matmul(activationHidden1, weights)+biases)

    with tf.name_scope("softmax_linear"):
        weights = tf.Variable(tf.truncated_normal([hiddenUnits2, numberClasses], stddev=1.0/tf.sqrt(float(hiddenUnits2)),
                              name="weights"))
        biases = tf.Variable(tf.zeros([numberClasses]), name = "biases")

        finalLogits = tf.matmul(activationHidden2, weights)+biases

    return finalLogits

def mnist_loss(labels, logits):
    """
        args:
            labels and logits (predicted labels)
        returns
            The loss function value
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits));
    
    return loss

def mnist_training(xData, yData, learningRate, dimension, hiddenUnits1, hiddenUnits2, numberClasses):
    """
        args:
            xData, yData : placeholders for the images data and the labels
            learningRate : value of the learning rate
            dimension : size of the images
            hiddenUnits1 : Size of the first layer of hidden units
            hiddenUnits2 : Size of the second layer of hidden units
            numberClasses : number of classes to classify
        returns
            train_step : The training operation with Gradient descent
            loss : The tensor of the loss
            logits : The tensor of the logits
    """
    logits = mnist_inferenceGraph(xData, dimension, hiddenUnits1, hiddenUnits2, numberClasses)
    loss = mnist_loss(yData, logits)

    #Gradient descent optimizer
    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    return train_step, loss, logits

def main():
    """
        Main function to execute the code
    """
    #Models sizes
    NUMBER_CLASSES = 10

    WIDTH_IMAGE = 28
    HEIGHT_IMAGE = 28
    DIMENSION = WIDTH_IMAGE*HEIGHT_IMAGE

    BATCH_SIZE = 100

    #Model parameters
    HIDDEN_UNITS1 = 128
    HIDDEN_UNITS2 = 32

    TRAINING_STEPS = 10000

    #Load Data
    TRAIN_DIRECTORY = "MNIST_data"
    mnist = read_data_sets(TRAIN_DIRECTORY, one_hot = True)

    #Data placeholders
    xData = tf.placeholder(tf.float32, [None, DIMENSION])
    yData = tf.placeholder(tf.float32, [None, NUMBER_CLASSES]) #One hot representation

    train_step, loss, logits = mnist_training(xData, yData, 0.01, DIMENSION, HIDDEN_UNITS1, HIDDEN_UNITS2, NUMBER_CLASSES)

    #Accuracy
    #Computes the accuracy
    yPredicted = tf.nn.softmax(logits)
    correctPrediction = tf.equal(tf.argmax(yData, axis=1),tf.argmax(yPredicted, axis=1))
    testAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    
    #Add checkpoints to save progress in training
    checkpointDir = "./checkpoints"
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
    
    #Save and store tutorial : https://github.com/nlintz/TensorFlow-Tutorials/blob/master/10_save_restore_net.ipynb
    #Create a global step variable and a saver
    #!! Saver MUST be declared after all variables otherwise the variable will not be saved
    globalStep = tf.Variable(0, name="globalStep", trainable=False)
    saver = tf.train.Saver()

    #Tensorflow session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    #Start training from the value saved in global step
    checkpoint = tf.train.get_checkpoint_state(checkpointDir)

    #If the checkpoint exist, restores from that
    if checkpoint and checkpoint.model_checkpoint_path:
        print(checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    start = globalStep.eval()

    print("Starting training from step %f", start)

    #Apply gradient descent
    lossValues = []
    for i in range(start, TRAINING_STEPS):
        xBatch, yBatch = mnist.train.next_batch(BATCH_SIZE)
        _, currentLoss = sess.run([train_step, loss], {xData:xBatch, yData:yBatch})
        lossValues.append(currentLoss)
        globalStep.assign(i).eval()
        
        #Only save every N step otherwise takes a lot of space on disk
        if(i%1000 == 0):
            saver.save(sess, checkpointDir + "/model.ckpt", global_step = globalStep)
            print("Accuracy : %f ", sess.run([testAccuracy], {xData: mnist.test.images, yData:mnist.test.labels}))

    plt.plot(lossValues)
    plt.show()


if __name__ == "__main__":
    main()

