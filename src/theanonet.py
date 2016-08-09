from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv1DLayer
from lasagne.layers import MaxPool1DLayer
from lasagne.nonlinearities import sigmoid, softmax
import theano
import numpy as np
from nolearn.lasagne.base import TrainSplit, BatchIterator
from nolearn.lasagne import NeuralNet
import theano.tensor.shared_randomstreams
import lasagne
from lasagne.updates import adam, nesterov_momentum
import theano.tensor as T
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.objectives import aggregate
from lasagne.layers import get_output


def float32(k):
    '''
    Cast a number to a single precision (32bit)
    floating point number.
    :param k: The number to cast
    :return: The floating point representation
    of the number in 32 bits.
    '''
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100, verbose=False):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.verbose = verbose

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if self.verbose:
                print ("Early stopping. Best valid loss was {:.6f} at epoch {}.\n".format(
                    self.best_valid, self.best_valid_epoch))
            if self.best_weights is not None:
                nn.load_params_from(self.best_weights)
            raise StopIteration()

class DNNClassifier:

    def __init__(self, length, num_channels,  eval_size, hidden_layers, dropout_layers, learning_rate=0.001, momentum=0.8, patience=100,
                 max_epochs=2, batch_size=128, verbose=False, seed=12345):
        '''
        :param num_inputs: Number of input units
        :param eval_size: The size [0, 1.0) which should be used for the validation size.
        :param layers: A python list describing the hidden layers, e.g. [4000,2000,1000]
        :param dropout_layers: A python list describing the dropout params from visible to hidden,
        e.g. [0,0.25,0.25,0.25]
        :param learning_rate: The initial learning rate
        :param momentum: The initial momentum rate
        :param patience: The number of iterations the optimization algorithm is allowed to run
        without improvement.
        :param max_epochs: Maximum number of iterations to run the optimization algorithm.
        :param batch_size: Size of each mini-batch
        :param verbose: Print debug information about the optimization process?
        :param seed: Number used to initialize the random number generator
        '''

        np.random.seed(seed)
        lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(seed)

        layer = InputLayer(shape=(None, num_channels, length), name='Input')
        #layer = DropoutLayer(layer, p=dropout_layers[0], name='Dropout_input')

        layer = Conv1DLayer(layer, 15, 15)
        layer = MaxPool1DLayer(layer, 35, 1)

        for lidx, num_units in enumerate(hidden_layers):
            layer = DenseLayer(layer, num_units=num_units, name='Dense_%d' % (lidx+1))
            layer = DropoutLayer(layer, p=dropout_layers[lidx], name='Dropout_%d' % (lidx+1))
        layer = DenseLayer(layer, num_units=2, nonlinearity=softmax, name='Output')

        clf = NeuralNet(layers=layer,
                        regression=False,
                        update=nesterov_momentum,
                        update_learning_rate=theano.shared(float32(learning_rate)),
                        update_momentum=theano.shared(float32(momentum)),
                        on_epoch_finished=[
                            AdjustVariable('update_learning_rate', start=learning_rate, stop=0.0001),
                            AdjustVariable('update_momentum', start=momentum, stop=0.999),
                            EarlyStopping(patience=patience, verbose=verbose),
                        ],
                        verbose=1 if verbose else 0,
                        max_epochs=max_epochs,
                        train_split=TrainSplit(eval_size=eval_size),
                        batch_iterator_train=BatchIterator(batch_size=batch_size)
                        )
        self.clf = clf

    def fit(self, X, y):
        '''
        :require X.dtype == np.float32
        :require y.dtype == np.int32
        :param X: Input matrix
        :param y: Output vector
        '''
        return self.clf.fit(X, y)

    def predict(self, X):
        '''
        :require X.dtype == np.float32
        :param X: Input matrix
        '''
        return self.clf.predict(X)
