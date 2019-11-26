import numpy as np
import pickle

DATA_FILE = 'checkpoints/data.pkl'

class Dataset:
    def __init__(self,
                 images_train,
                 responses_train,
                 images_val,
                 responses_val,
                 images_test,
                 responses_test):

        # normalize images (mean=0, SD=1)
        m = images_train.mean()
        sd = images_train.std()
        zscore = lambda img: (img - m) / sd
        self.images_train = zscore(images_train)[...,None]
        self.images_val = zscore(images_val)[...,None]
        self.images_test = zscore(images_test)[...,None]
        
        # normalize responses (SD=1)
        sd = responses_train.std(axis=0)
        sd[sd < (sd.mean() / 100)] = 1
        def rectify_and_normalize(x):
            x[x < 0] = 0    # responses are non-negative; this gets rid
                            # of small negative numbers due to numerics
            return x / sd
        self.responses_train = rectify_and_normalize(responses_train)
        self.responses_val = rectify_and_normalize(responses_val)
        self.responses_test = rectify_and_normalize(responses_test)
        
        self.num_neurons = responses_train.shape[1]
        self.num_train_samples = images_train.shape[0]
        self.px_x = images_train.shape[2]
        self.px_y = images_train.shape[1]
        self.input_shape = [None, self.px_y, self.px_x, 1]
        self.minibatch_idx = 1e10
        self.train_perm = []

    def introspection(self):

        print("# Training images")
        print(type(self.images_train))
        print(self.images_train.shape)
        print(self.images_train.dtype)
        print(np.min(self.images_train))
        print(np.mean(self.images_train))
        print(np.median(self.images_train))
        print(np.max(self.images_train))
        print(np.std(self.images_train))

        print("# Training responses")
        print(type(self.responses_train))
        print(self.responses_train.shape)
        print(self.responses_train.dtype)
        print(np.min(self.responses_train))
        print(np.mean(self.responses_train))
        print(np.median(self.responses_train))
        print(np.max(self.responses_train))
        # print(self.responses_train)
        # print(self.responses_train[0])

        print("# Testing images")
        print(type(self.images_test))
        print(self.images_test.shape)
        print(self.images_test.dtype)
        print(np.min(self.images_test))
        print(np.mean(self.images_test))
        print(np.median(self.images_test))
        print(np.max(self.images_test))
        print(np.std(self.images_test))

        print("# Testing responses")
        print(type(self.responses_test))
        print(self.responses_test.shape)
        print(self.responses_test.dtype)
        print(np.min(self.responses_test))
        print(np.mean(self.responses_test))
        print(np.median(self.responses_test))
        print(np.max(self.responses_test))
        # print(self.responses_test)
        # print(self.responses_test[0])

        print("# Validation images")
        print(type(self.images_val))
        print(self.images_val.shape)
        print(self.images_val.dtype)
        print(np.min(self.images_val))
        print(np.mean(self.images_val))
        print(np.median(self.images_val))
        print(np.max(self.images_val))
        print(np.std(self.images_val))

        print("# Validation responses")
        print(type(self.responses_val))
        print(self.responses_val.shape)
        print(self.responses_val.dtype)
        print(np.min(self.responses_val))
        print(np.mean(self.responses_val))
        print(np.median(self.responses_val))
        print(np.max(self.responses_val))
        # print(self.responses_val)
        # print(self.responses_val[0])

        print("num neurons: {}".format(self.num_neurons))
        print("num train samples: {}".format(self.num_train_samples))
        print("px x: {}".format(self.px_x))
        print("px y: {}".format(self.px_y))
        print("input shape: {}".format(self.input_shape))
        print("minibatch idx: {}".format(self.minibatch_idx))
        print("train_perm: {}".format(self.train_perm))

        return

    def val(self):
        return self.images_val, self.responses_val

    def train(self):
        return self.images_train, self.responses_train

    def test(self, averages=True):
        responses = self.responses_test.mean(axis=0) if averages else self.responses_test
        return self.images_test, responses

    def minibatch(self, batch_size):
        if self.minibatch_idx + batch_size > self.num_train_samples:
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return self.images_train[idx, :, :], self.responses_train[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)

    @staticmethod
    def load(data_file=DATA_FILE):
        with open(data_file, 'rb') as file:
            return pickle.load(file)
