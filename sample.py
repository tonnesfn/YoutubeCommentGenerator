import tensorflow as tf
import numpy as np
import random
import RNN


class SampleGenerator:
    x = None
    y = None

    saver = None

    model = None
    network = None

    directory = 'output/201707081558'

    def get_prediction(self):

        sequence = np.array([[0] * (len(self.model.dataset.dictionary) + 1)] * self.model.n_steps)
        for i in range(len(sequence)):
            sequence[i][random.randint(0, len(self.model.dataset.dictionary))] = 1

        sequence[-1] = np.array([[0] * (len(self.model.dataset.dictionary) + 1)])

        print('Running tensorflow in CPU mode')
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:
            self.saver.restore(sess, "output/201707081114/models/model-7-3928.ckpt")

            for iteration in range(300):
                x_batch = np.array(sequence[-self.model.n_steps:]).reshape(1, self.model.n_steps, self.model.n_inputs)
                y_pred = sess.run(self.network, feed_dict={self.x: x_batch})

                sequence = np.vstack([sequence, y_pred[-1]])

            output_string = RNN.dataset.decode(sequence[-300:])

            return output_string

    def print_sample(self):
        print(self.get_prediction())

    def __init__(self):
        self.model = RNN.Model()
        self.model.restore_dataset(self.directory)

        self.x = tf.placeholder('float', [None, self.model.n_steps, self.model.n_inputs])
        self.y = tf.placeholder('float', [None, self.model.n_steps, self.model.n_outputs])

        self.network = self.model.recurrent_neural_network(self.x)

        self.saver = tf.train.Saver()

if __name__ == "__main__":
    sampleGenerator = SampleGenerator()
    sampleGenerator.print_sample()
