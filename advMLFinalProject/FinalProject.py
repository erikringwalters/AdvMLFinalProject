import tensorflow as tf
import numpy as np
tf.reset_default_graph()             

data = open('../alice.txt', 'r').read() # should be simple plain text file

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
def fetchBatch(X,y,batch_size, num_batches):
    for i in range(num_batches):
        idx = np.random.randint((len(X)-batch_size)-1)
        X_batch = X[idx:idx+batch_size]
        y_batch = y[idx:idx+batch_size]
        yield X_batch, y_batch

# Pulls a single random row from X_modifed
def single_batch(X):
    rnd = np.random.randint(len(X)-1)
    return X[rnd]
    


# @params: seed_id: a random indicie to seed the RNNs output
# @params: n: the number of characters to generate in total
    
n_inputs = vocab_size
n_outputs = vocab_size
n_steps = 25
n_neurons = 100

X = tf.placeholder(tf.int32, [None, n_steps])
one_hot_X = tf.one_hot(X, vocab_size, dtype=tf.float32)
y = tf.placeholder(tf.int32, [None, n_steps])

layers = tf.contrib.rnn.OutputProjectionWrapper(
         tf.contrib.rnn.BasicRNNCell(num_units=n_neurons),
         output_size=vocab_size, activation=tf.nn.tanh)


brnn_layers = [tf.contrib.rnn.BasicRNNCell(size) for size in [n_neurons,250]]
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.MultiRNNCell(brnn_layers),output_size=vocab_size, activation=tf.nn.tanh)
output, states = tf.nn.dynamic_rnn(cell, one_hot_X, dtype=tf.float32) # this takes size [None, n_steps, n_inputs]
learning_rate = 0.01

y_probs = tf.nn.softmax(output)
xenthropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
loss = tf.reduce_mean(xenthropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

def sample(seed_pred, n):
    start_idx = 0
    # list of indexes we predict
    output = []
    # Debug for seeing the prob distributions
    
    for i in range(n):
      # This creates the moving window that starts with our input sequence, and the new input is appended at each loop
      window = seed_pred[start_idx:start_idx + n_steps]
      # We then rotate it to batch_size = 1 and 25 cols for entry into the X tensor
      window = window.reshape((1, n_steps))
      # We evaluate the softmax of the outputs by inserting our array window into the network and getting the logits
      char_prob = y_probs.eval(feed_dict={X:window})
      # We only care about the last character softmaxed outputs, as this is the probability of a choice of one of len(vocab_size) characters
      last_char = char_prob[0][n_steps-1]
      # We get an index value by pulling a random value /according to the probabilty distribution of the last_chars softmax      
      ind = np.random.choice(range(vocab_size), p=last_char.ravel())
      # tack that onto our big list
      output.append(ind)
      # reshape ind so we can tack it to the bottom of seed_pred
      ind = ind.reshape((1,))
      # append it to seed_pred
      seed_pred = np.append(seed_pred, ind, axis=0)
      # move the window up one and continue
      start_idx +=1
    return output






# FIXME add accuracy, refer to Geron pg 397
init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 25
num_batches = 50
losses = []
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in fetchBatch(X_modified,y_modified,batch_size,50):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if epoch % 10 == 0:
            cur_loss = loss.eval(feed_dict={X:X_batch, y:y_batch})
            losses.append(cur_loss)
            print(f"Loss: \t{cur_loss}")
            solo_batch = single_batch(X_modified)
            smpl = sample(solo_batch,25)
            txt = ''.join(ix_to_char[ix] for ix in smpl)
            print('----OUTPUT \n %s \n----' % (txt, ))
                
                    

tf.reset_default_graph()             


