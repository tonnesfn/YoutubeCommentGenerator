import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

import gen_dataset

import time


class Model:

    dataset = gen_dataset.Dataset()

    # 64: 152/s, 128: 178/s, 512: 198/s
    batch_size = 512
    rnn_size = 512
    hm_epochs = 20

    n_classes = None
    n_inputs = None
    n_outputs = None
    n_steps = None

    x = None
    y = None

    def recurrent_neural_network(self):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        self.x = tf.transpose(x, [1, 0, 2])  # 300 * [?, 96]
        self.x = tf.reshape(x, [-1, self.n_inputs])
        self.x = tf.split(x, self.n_steps, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = []

        for i in range(len(outputs)):
            output.append(tf.matmul(outputs[i], layer['weights']) + layer['biases'])

        output_return = tf.reshape(tf.concat(axis=1, values=output), [-1, self.n_inputs], name='network_output')  # [?, 96]

        return output_return

    def train_neural_network(self, x, saved_state_file):

        with tf.Session() as sess:

            if len(saved_state_file) == 0:
                prediction = self.recurrent_neural_network(self.x)
                saver = tf.train.Saver()

                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
                optimizer = tf.train.AdamOptimizer().minimize(cost)
            else:
                directory = '/'.join(cont_file.split('/')[0:-1])
                saver = tf.train.import_meta_graph(saved_state_file + '.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint(directory))

                self.x = tf.get_default_graph().get_tensor_by_name('x:0')
                x = self.x
                self.y = tf.get_default_graph().get_tensor_by_name('y:0')
                prediction = tf.get_default_graph().get_tensor_by_name('network_output:0')

                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
                optimizer = tf.get_default_graph().get_operation_by_name('Adam')

            sess.run(tf.global_variables_initializer())

            sample_counter = 0
            loop_counter = 0
            start_time = time.time()
            for epoch in range(self.hm_epochs):
                epoch_loss = 0

                for _ in range(int(self.dataset.num_examples / self.batch_size)):
                    epoch_x, epoch_y = self.dataset.next_batch(self.batch_size)

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                    sample_counter += self.batch_size
                    loop_counter += 1

                    if loop_counter == 100:
                        current_time = time.time()
                        print('Time: {}samples/s - {}m/epoch'.format(
                            int(sample_counter / (current_time - start_time)),
                            int((self.dataset.num_examples / (int(sample_counter / (current_time - start_time))))/60)
                            ), end='\r')

                        loop_counter = 0

                save_path = saver.save(sess, self.dataset.current_directory + '/models/model-{}-{}.ckpt'.format(epoch, int(epoch_loss)))

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)
                print('Model saved in file: ', save_path)

                self.dataset.current_batch_index = 0  # Reset batch index before next loop

    def train(self, cont_file):
        self.train_neural_network(self.x, cont_file)

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

        #self.x = tf.placeholder('float', [None, self.n_steps, self.n_inputs], name='x')
        #self.y = tf.placeholder('float', [None, self.n_steps, self.n_outputs], name='y')

    def init_new_dataset(self):
        self.dataset.generate_new_dataset()
        self.init_dataset()

    def restore_dataset(self, directory):
        self.dataset.restore_dataset(directory)
        self.init_dataset()

if __name__ == "__main__":
    cont_file = 'output/201707090024/models/model-10-868'
    #cont_file = ''
    cont_file_directory = '/'.join(cont_file.split('/')[0:-1])
    output_directory = '/'.join(cont_file_directory.split('/')[0:-1])

    model = Model()

    if len(cont_file) > 0:
        model.restore_dataset(output_directory)
    else:
        model.init_new_dataset()

    model.train(cont_file)
