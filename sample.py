import tensorflow as tf
import numpy as np
import RNN

x = tf.placeholder('float', [None, RNN.n_steps, RNN.n_inputs])
y = tf.placeholder('float', [None, RNN.n_steps, RNN.n_outputs])

prediction = RNN.recurrent_neural_network(x)

saver = tf.train.Saver()

n_steps = 300
n_inputs = RNN.n_inputs

sequence = [[0] * len(RNN.dataset.dictionary) + [1]] * n_steps

with tf.Session() as sess:
    saver.restore(sess, "output/save_model/model-4-1666.ckpt")

    for iteration in range(300):
        x_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, n_inputs)
        y_pred = sess.run(prediction, feed_dict={x: x_batch})

        sequence.append(y_pred[-1])

    print("Model restored.")
