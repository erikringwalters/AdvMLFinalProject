import tensorflow as tf
import numpy as np

data = open('E:/Documents/beemovie.txt', 'r').read() # should be simple plain text file

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

# @params: X: the input array to fetch a batch from
# @params: y the output array to fetch a corosponding batch from
# @params: batch_size: The size of the batch you want to pull
def fetchBatch(X,y,batch_size):
    idx = np.random.randint((len(X)-batch_size)-1)
    X_batch = X[idx:idx+batch_size,:]
    y_batch = X[idx:idx+batch_size,:]
    yield X_batch, y_batch


# @params: seed_id: a random indicie to seed the RNNs output
# @params: n: the number of characters to generate in total

    
    
    
    
n_inputs = vocab_size
n_outputs = vocab_size
n_steps = 25
n_neurons = 100



X = tf.placeholder(tf.int32, [None, n_steps])
# in line one hot encoding in tf, we tried to one hot everything then pass it through but had trouble creating a fetch batch for it
one_hot_X = tf.one_hot(X, vocab_size)
y = tf.placeholder(tf.int32, [None, n_steps])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(
        num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs)
    
outputs, states = tf.nn.dynamic_rnn(cell, one_hot_X, dtype=tf.float32) # this takes size [None, n_steps, n_inputs]
learning_rate = 0.001

y_probs = tf.nn.softmax(logits=outputs)
xenthropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=y)
loss = tf.reduce_mean(xenthropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

tf.reset_default_graph()

# FIXME add accuracy, refer to Geron pg 397
init = tf.global_variables_initializer()

n_epochs = 1000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in fetchBatch(X_modified,y_modified,batch_size):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            if epoch % 100 == 0:
                cur_loss = loss.eval(feed_dict={X:X_batch, y:y_batch})
                print(f"Loss: \t{cur_loss}")
                for single_batch, _ in fetchBatch(X_modified,y_modified,1):
                    probs = y_probs.eval(feed_dict={X:single_batch})
                    
                



