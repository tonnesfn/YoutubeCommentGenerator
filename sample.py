import tensorflow as tf
import numpy as np
import RNN
import random

x = tf.placeholder('float', [None, RNN.n_steps, RNN.n_inputs])
y = tf.placeholder('float', [None, RNN.n_steps, RNN.n_outputs])

prediction = RNN.recurrent_neural_network(x)

saver = tf.train.Saver()

n_steps = 300
n_inputs = RNN.n_inputs

sequence = np.array([[0] * (len(RNN.dataset.dictionary) + 1)] * n_steps)
for i in range(len(sequence)):
    sequence[i][random.randint(0, len(RNN.dataset.dictionary))] = 1

sequence[-1] = np.array([[0] * (len(RNN.dataset.dictionary) + 1)])

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

with tf.Session(config=config) as sess:
    saver.restore(sess, "output/save_model/model-0-4952.ckpt")

    for iteration in range(300):
        x_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, n_inputs)
        y_pred = sess.run(prediction, feed_dict={x: x_batch})

        sequence = np.vstack([sequence, y_pred[-1]])

    output_string = RNN.dataset.decode(sequence[-300:])

    print(output_string)
