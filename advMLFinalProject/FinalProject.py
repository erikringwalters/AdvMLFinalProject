# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:43:02 2018

@author: Erik
"""

import tensorflow as tf
import numpy as np

data = open('./Documents/beemovie.txt', 'r').read() # should be simple plain text file

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

seq_length = 25

X = []
y = []

for i in range(0, len(data)-seq_length-1, 1):
        X.append([char_to_ix[ch] for ch in data[i:i+seq_length]])
        y.append([char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])

# reshape the data
# in X_modified, each row is an encoded sequence of characters
X_modified = np.reshape(X, (len(X), seq_length))
y_modified = np.reshape(y, (len(y), seq_length))

# n_steps is just the sequence length
# we need to build a batch function that gives us batches of size [mini_batch_size,n_steps,n_inputs]
n_inputs = 1
n_steps = 25
n_neurons = 100
# give one hot numeric strings

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, vocab_size])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

init = tf.global_variables_initializer()

oneHot = tf.one_hot(X_modified[0,:],vocab_size)
with tf.Session() as sess:
    init.run()
    one_hot_out = oneHot.eval()
    print(one_hot_out)