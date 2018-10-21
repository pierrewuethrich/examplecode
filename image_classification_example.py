# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:04:27 2018

@author: Wuethrich Pierre

This is some example code of image classification using a simple CNN, 
using the MNIST data (hand-written digits)

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# Getting the MNIST data-set
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#------------------------------------------------------------------
# Creating functions which will help readability/structure of code


# Initializing the weights of the NN-layers

def init_weights(shape):
    init_rand_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_rand_dist)

# Initializing the bias-terms

def init_bias(shape):
    init_bias_values = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_values)

# Defining the two-dimenstional convolution and pooling functions
# using tf built-in conv2d and max_pool methods
    
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
# Defining the convolutional layers and normal layers

def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b= init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

#-----------------------------------------------------------------------
# Defining the Placeholders, Layers, Optimizer, Loss-function,etc
    
# Placeholders
    
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])

# Layers (2 times convolution and 2 x subpooling)

x_image = tf.reshape(x,[-1,28,28,1])

convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# Flattening of the output
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_1 = normal_full_layer(convo_2_flat,1000)

# Introducing node drop-out and getting y_pred

holding_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_1,keep_prob=holding_prob)

y_pred = normal_full_layer(full_one_dropout,10)

# Defining the loss-function and optimizer

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# Initialization of the variables and Session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    steps = 500
    
    for i in range(steps):
        
        batch_x , batch_y = mnist.train.next_batch(50)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,holding_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%50 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,holding_prob:1.0}))
            print('\n')



