import json
import random
import datetime
import os
import numpy as np


class Dataset:
    current_batch_index = 0
    comments = []
    dictionary = {}
    longest_comment = 0
    num_examples = 0
    max_comment_length = 300
    current_directory = ''

    # Decodes a given one hot encoded list to a string
    def decode(self, encoded_list):
        decoded_string = ''

        for character in encoded_list:
            max_index = character.argmax()
            c = [k for (k, v) in self.dictionary.items() if v == max_index]

            if max_index == len(self.dictionary):
                decoded_string += '§'
            else:
                decoded_string += c[0]

        return decoded_string

    def gen_dict(self, directory):
        print('Generated new dictionary:')

        self.dictionary = dict.fromkeys(''.join(self.comments), 0)

        index = 0
        for x in self.dictionary:
            self.dictionary[x] = index
            index = index + 1

        self.longest_comment = len(max(self.comments, key=len))
        self.num_examples = len(self.comments)

        self.write_dict(directory, self.dictionary)

    # todo: make new folder if it does not exist
    def write_dict(self, directory, dictionary):
        with open(directory + '/dict.json', 'w') as f:
            json.dump(dictionary, f, indent=4)

        settings = {'longest_comment': self.longest_comment, 'num_examples': self.num_examples,
                    'max_comment_length': self.max_comment_length}

        with open(directory + '/settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    def read_dict(self, filename):
        with open(filename, 'r') as f:
            try:
                self.dictionary = json.load(f)
            except ValueError:
                print('Error reading dictionary JSON file!')
                self.dictionary = {}

        with open(self.current_directory + '/settings.json', 'r') as f:
            try:
                settings = json.load(f)

                self.longest_comment = settings['longest_comment']
                self.num_examples = settings['num_examples']
                self.max_comment_length = settings['max_comment_length']

                print('Read settings file: longest: {}, num_examples: {}, max_comment_length: {}'.
                      format(self.longest_comment, self.num_examples, self.max_comment_length))
            except ValueError:
                print('Error reading settings JSON file!')

    def get_comments(self, filename):
        with open(filename) as f:
            content = f.readlines()

        self.comments = [x.strip() for x in content]

        for i in range(len(self.comments)):
            self.comments[i] = self.comments[i][:self.max_comment_length]

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

            # Pad until you reach full string length
            for j in range(self.longest_comment - len(x[i]) + 1):
                line.append(([0] * (len(self.dictionary))) + [1])  # Space for full dictionary plus padding

            features.append(line[:-1])
            labels.append(line[1:])

        features = np.array(features)
        labels = np.array(labels)

        self.current_batch_index += 1
        return features, labels

    def makeDirectory(self, directory_name):
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    def generate_new_dataset(self):
        now = datetime.datetime.now()
        current_directory = 'output/{:04d}{:02d}{:02d}{:02d}{:02d}'.\
            format(now.year, now.month, now.day, now.hour, now.minute)

        self.makeDirectory(current_directory)
        self.makeDirectory(current_directory+'/models')

        self.get_comments('data/mergedComments.txt')

        self.gen_dict(current_directory)

        random.shuffle(self.comments)

    def restore_dataset(self, directory):
        self.current_directory = directory
        self.read_dict(self.current_directory + '/dict.json')

