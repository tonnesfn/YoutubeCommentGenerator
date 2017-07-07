import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

import gen_dataset

# Try dynamic_rnn since we don't want to give the sequence length
# Zero pad within each batch - find examples
# Sparse cross entropy with ints instead of one hot encoded outputs?

dataset = gen_dataset.Dataset()

hm_epochs = 5
n_classes = len(dataset.dictionary)+1
batch_size = 8
n_inputs = len(dataset.dictionary)+1  # Chunk_size
n_outputs = len(dataset.dictionary)+1
n_steps = dataset.longest_comment  # n_chunks
rnn_size = 128

print('Dictionary size: {}'.format(len(dataset.dictionary)))
print('Longest comment: {}'.format(dataset.longest_comment))

x = tf.placeholder('float', [None, n_steps, n_inputs])
y = tf.placeholder('float', [None, n_steps, n_outputs])


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])  # 300 * [?, 96]
    x = tf.reshape(x, [-1, n_inputs])
    x = tf.split(x, n_steps, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = []

    for i in range(len(outputs)):
        output.append(tf.matmul(outputs[i], layer['weights']) + layer['biases'])

    output_return = tf.reshape(tf.concat(axis=1, values=output), [-1, n_inputs])  # [?, 96]

    return output_return


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(dataset.num_examples / batch_size)):
                epoch_x, epoch_y = dataset.next_batch(batch_size)

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            dataset.current_batch_index = 0  # Reset batch index before next loop

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

train_neural_network(x)
