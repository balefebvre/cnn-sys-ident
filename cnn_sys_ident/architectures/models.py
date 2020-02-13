import datetime
import numpy as np
import os
import tensorflow as tf


class TFSession:

    _saver = None

    @property
    def saver(self):
        if self._saver is None:
            with self.graph.as_default():
                self._saver = tf.compat.v1.train.Saver(max_to_keep=1)
        return self._saver

    def __init__(self, log_dir=None, log_hash=None):
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        if log_hash is None:
            log_hash = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, log_hash)
        self.log_hash = log_hash
        self.log_checkpoints = os.path.join(self.log_dir, "checkpoints", "model.ckpt")
        self.log_tensorboard = os.path.join(self.log_dir, "tensorboard")
        self.seed = int.from_bytes(log_hash[:4].encode('utf8'), 'big')
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(self.seed)
            np.random.seed(self.seed)
        self.session = tf.compat.v1.Session(graph=self.graph)

    def __del__(self):
        try:
            self.close()
        except tf.errors.OpError:
            pass

    def close(self):
        if self.session is not None:
            self.session.close()

    def save(self):
        with self.graph.as_default():
            self.saver.save(self.session, self.log_checkpoints)

    def load(self):
        with self.graph.as_default():
            from tensorflow.python.util import deprecation
            deprecation._PRINT_DEPRECATION_WARNINGS = False
            self.saver.restore(self.session, self.log_checkpoints)
            deprecation._PRINT_DEPRECATION_WARNINGS = True


class BaseModel:
    def __init__(self, data, log_dir=None, log_hash=None):
        self.tf_session = TFSession(log_dir=log_dir, log_hash=log_hash)
        self.data = data
        with self.tf_session.graph.as_default():
            self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
            self.inputs = tf.compat.v1.placeholder(tf.float32, shape=data.input_shape, name='inputs')
            self.responses = tf.compat.v1.placeholder(tf.float32, shape=[None, data.num_neurons], name='responses')

    def evaluate(self, var, *args, **kwargs):
        return self.tf_session.session.run(var, *args, **kwargs)

    def load(self):
        self.tf_session.load()


class CorePlusReadoutModel:
    def __init__(self, base, core, readout):
        self.base = base
        self.core = core
        self.readout = readout
        self.predictions = readout.output

    def load(self):
        self.base.load()
