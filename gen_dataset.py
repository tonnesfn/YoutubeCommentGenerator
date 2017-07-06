import json
import random
import numpy as np

class Dataset:
    current_batch_index = 0
    comments = []
    dictionary = {}
    longest_comment = 0
    num_examples = 0

    def gen_dict(self):
        self.dictionary = dict.fromkeys(''.join(self.comments), 0)

        index = 0
        for x in self.dictionary:
            self.dictionary[x] = index
            index = index + 1
        self.write_dict('output/dict.json', self.dictionary)

    # todo: make new folder if it does not exist
    def write_dict(self, filename, dictionary):
        with open(filename, 'w') as f:
            json.dump(dictionary, f)

    def read_dict(self, filename):
        with open(filename, 'r') as f:
            try:
                self.dictionary = json.load(f)
            except ValueError:
                self.dictionary = {}

    def getComments(self, filename):
        with open(filename) as f:
            content = f.readlines()
        self.comments = [x.strip() for x in content]

    # Return batch_size number of examples and labels
    def next_batch(self, batch_size):
        start = min(self.current_batch_index * batch_size, len(self.comments))
        end = min(((self.current_batch_index+1) * batch_size), len(self.comments))

        # Jeg skal ha ut en array av batch_size av kommentarer som hver av de er one-hot encodet chars
        x = self.comments[start:end]
        y = self.comments[start:end]

        features = []
        labels = []

        # For each comment in current batch:
        for i in range(end - start):
            line = []

            # For each character in the comment:
            for j in range(len(x[i])):
                one_hot_encoding = [0] * (len(self.dictionary) + 1)  # Space for full dictionary plus padding
                one_hot_encoding[self.dictionary[x[i][j]]] = 1
                line.append(one_hot_encoding)

            for j in range(self.longest_comment - len(x[i])):
                line.append(([0] * (len(self.dictionary))) + [1])  # Space for full dictionary plus padding

            features.append(line[:-1])
            labels.append(line[1:])

        # # Debug writing to file:
        # f = open('batch_debugging_inputs.txt', 'w')
        # f.writelines(["%s\n" % item for item in self.comments[start:end]])
        # f.close()
        #
        # f = open('batch_debugging_result.txt', 'w')
        # f.writelines(["%s\n" % item for item in features])
        # f.close()

        features = np.array(features)
        labels = np.array(labels)

        self.current_batch_index += 1
        return features, labels

    def __init__(self):
        self.getComments('data/mergedComments.txt')
        random.shuffle(self.comments)
        self.gen_dict()
        self.longest_comment = len(max(self.comments, key=len))
        self.num_examples = len(self.comments)

#dataset = Dataset()
#dataset.comments = dataset.comments[:11]
