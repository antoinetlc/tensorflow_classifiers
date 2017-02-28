import tensorflow as tf
import numpy as np
import pylab

#Generate my data
xData = np.random.rand(100).astype(np.float32) #Random numbers in 0;1
noise = np.random.normal(scale=0.01, size=len(xData)) #Random gaussian distributed with std deviation 0.01
yData = 0.1*xData+0.4+noise

#Model
W = tf.Variable(tf.random_uniform([1], 0.0, 1.0))
b = tf.Variable(tf.random_uniform([1], 0.0, 1.0))
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linearModel = W*x+b

#loss
loss = tf.reduce_sum(tf.square(linearModel-y))
optimizer = tf.train.GradientDescentOptimizer(0.005) #If nan appears : decrease the learning rate
train = optimizer.minimize(loss)

#Tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Optimise
W_initial, b_initial = sess.run([W, b], {x:xData, y:yData})
initialGuess = W_initial*xData+b_initial

print(sess.run([W, b, loss], {x:xData, y:yData}))

for i in range(1000):
    sess.run(train, {x:xData, y:yData})


W_final, b_final = sess.run([W, b], {x:xData, y:yData})
finalGuess = W_final*xData+b_final


pylab.plot(xData, yData, '.', label='Measurements')
pylab.plot(xData, initialGuess, '.', label='Initial Guess')
pylab.plot(xData, finalGuess, '.', label='Final Guess')
pylab.legend()
pylab.show()