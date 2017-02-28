import tensorflow as tf
import numpy as np
import pylab
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#Parameters
W = tf.Variable(tf.random_uniform([784,10]),tf.float32)
b = tf.Variable(tf.random_uniform([10]), tf.float32)
x = tf.placeholder(tf.float32, [None, 784])

#Model
y_predicted = tf.matmul(x,W)+b #size 10*None
y = tf.placeholder(tf.float32, [None,10], name="y_labels")

#Numerically unstable formula
#crossEntropyLoss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(y_predicted), axis=0)) #element wise multiplication

crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predicted))

#Optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropyLoss)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

lossValues = []
for i in range(1000):
    batchX, batchY = mnist.train.next_batch(100)
    _, loss = sess.run([train_step, crossEntropyLoss], {x:batchX, y:batchY})
    lossValues.append(loss)


#Accuracy : count the number of correct preductions
correctPrediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_predicted, axis=1)) #boolean to be cast to floats
testAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

print(sess.run(testAccuracy, {x:mnist.test.images , y:mnist.test.labels}))

plt.plot(lossValues)
plt.show()
