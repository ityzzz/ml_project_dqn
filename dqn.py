import tensorflow as tf
import numpy as np
import random
from collections import deque

LEARNING_RATE = 0.01

LAYER_SIZE = 90

class DQN:
    def __init__(self, session, width, height, action_count, name):
        self.session = session
        self.width = width
        self.height = height
        self.name = name
        self.action_count = action_count
        
        self.input_size = self.width * self.height

        with tf.variable_scope(name):
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

            W1 = tf.get_variable("W1", shape=[self.input_size, LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.nn.relu(tf.matmul(self.X, W1))

            W2 = tf.get_variable("W2", shape=[LAYER_SIZE, LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.nn.relu(tf.matmul(L1, W2))

            W3 = tf.get_variable("W3", shape=[LAYER_SIZE, self.action_count], initializer=tf.contrib.layers.xavier_initializer())
            self.Qpredict = tf.matmul(L2, W3)

        self.Y = tf.placeholder(shape=[None, self.action_count], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Y - self.Qpredict))
        self.train = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def predict(self, state):
        a = self.session.run(self.Qpredict, feed_dict={self.X : state})
        return a

    def update(self, x_stack, y_stack):
        return self.session.run([self.loss, self.train], feed_dict={self.X : x_stack, self.Y : y_stack})