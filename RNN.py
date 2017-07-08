import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

import gen_dataset


class Model:

    dataset = gen_dataset.Dataset()

    batch_size = 64
    rnn_size = 512
    hm_epochs = 20

    n_classes = None
    n_inputs = None
    n_outputs = None
    n_steps = None

    x = None
    y = None

    def recurrent_neural_network(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1, 0, 2])  # 300 * [?, 96]
        x = tf.reshape(x, [-1, self.n_inputs])
        x = tf.split(x, self.n_steps, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = []

        for i in range(len(outputs)):
            output.append(tf.matmul(outputs[i], layer['weights']) + layer['biases'])

        output_return = tf.reshape(tf.concat(axis=1, values=output), [-1, self.n_inputs])  # [?, 96]

        return output_return

    def train_neural_network(self, x):
        prediction = self.recurrent_neural_network(x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                for _ in range(int(self.dataset.num_examples / self.batch_size)):
                    epoch_x, epoch_y = self.dataset.next_batch(self.batch_size)

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                save_path = saver.save(sess, self.dataset.current_directory + '/models/model-{}-{}.ckpt'.format(epoch, int(epoch_loss)))

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)
                print('Model saved in file: ', save_path)

                self.dataset.current_batch_index = 0  # Reset batch index before next loop

    def train(self):
        self.train_neural_network(self.x)

    def init_dataset(self):
        self.n_classes = len(self.dataset.dictionary) + 1
        self.n_inputs = len(self.dataset.dictionary) + 1
        self.n_outputs = len(self.dataset.dictionary) + 1
        self.n_steps = self.dataset.longest_comment

        print('Dataset initialized: \n'
              '    n_classes: {}\n'
              '    n_inputs: {}\n'
              '    n_outputs: {}\n'
              '    n_steps: {}'.format(self.n_classes, self.n_inputs, self.n_outputs, self.n_steps))

        self.x = tf.placeholder('float', [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder('float', [None, self.n_steps, self.n_outputs])

    def init_new_dataset(self):
        self.dataset.generate_new_dataset()
        self.init_dataset()

    def restore_dataset(self):
        self.dataset.restore_dataset()
        self.init_dataset()

if __name__ == "__main__":
    model = Model()
    model.init_new_dataset()
    model.train()
