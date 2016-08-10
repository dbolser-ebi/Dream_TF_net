import tensorflow as tf
import numpy as np
from sklearn.cross_validation import StratifiedKFold

import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKCYAN = '\033[36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_1xn(x, width):
    return tf.nn.max_pool(x, ksize=[1, 1, width, 1],
                          strides=[1, 1, width, 1], padding='VALID')


def calc_loss(logits, labels):
    entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(entropies)


class EarlyStopping:
    def __init__(self, max_stalls):
        self.best_loss = 100
        self.num_stalls = 0
        self.max_stalls = max_stalls

    def update(self, loss):
        if loss < self.best_loss:
            self.num_stalls = 0
            self.best_loss = loss
            return 2
        elif self.num_stalls < self.max_stalls:
            self.num_stalls += 1
            return 1
        else:
            return 0


class ConvNet:
    def __init__(self, model_dir, batch_size=256, num_channels=4, num_epochs=2,
                 width=200, num_outputs=1, eval_size=0.2, early_stopping=100, verbose=True):
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        self.height = 1
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.width = width
        self.eval_size = eval_size
        self.verbose = verbose
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          width, self.num_channels),
                                          name='features')
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs), name='labels')
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = self.get_model()
        self.early_stopping = EarlyStopping(max_stalls=early_stopping)
        self.transcription_factor = None

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def get_model(self):
        # Convolution
        with tf.variable_scope('conv1') as scope:
            depth_1 = 15
            width_1 = 15
            conv_kernel = weight_variable(shape=[self.height, width_1, self.num_channels, depth_1])
            conv_biases = weight_variable(shape=[depth_1])
            conv = conv1D(self.tf_sequence, conv_kernel)
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        # Pool
        with tf.variable_scope('pool1') as scope:
            pool_width_1 = 35
            pool1 = max_pool_1xn(conv1, pool_width_1)

        # Dense
        with tf.variable_scope('dense') as scope:
            fc_kernel = weight_variable([5 * depth_1, 100])
            fc_bias = bias_variable([100])

            flattened = tf.reshape(pool1, [-1, 5 * depth_1])
            fc1 = tf.nn.relu(tf.matmul(flattened, fc_kernel) + fc_bias)

        # Dropout
        with tf.variable_scope('dropout') as scope:
            drop = tf.nn.dropout(fc1, self.keep_prob)

        # Output
        with tf.variable_scope('output') as scope:
            fc_kernel2 = weight_variable([100, 1])
            fc_bias2 = bias_variable([1])
            logits = tf.matmul(drop, fc_kernel2) + fc_bias2

        return logits

    def fit(self, X, y):
        summary_writer = tf.train.SummaryWriter(self.model_dir + 'train')

        loss = calc_loss(self.logits, self.tf_train_labels)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        saver = tf.train.Saver()

        y_ = np.max(y, axis=1)
        kf = StratifiedKFold(y_, round(1. / self.eval_size))
        train_indices, valid_indices = next(iter(kf))
        X_train = X[train_indices]
        X_val = X[valid_indices]
        y_train = y[train_indices]
        y_val = y[valid_indices]

        print 'Train size', X_train.shape[0], 'Validation size', X_val.shape[0]

        with tf.Session() as session:
            tf.initialize_all_variables().run()

            if self.verbose:
                print
                print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

            # train model
            for epoch in xrange(1, self.num_epochs+1):
                start_time = time.time()
                # Training
                num_examples = X_train.shape[0]
                losses = []
                for offset in xrange(0, num_examples-self.batch_size, self.batch_size):
                    end = min(offset+self.batch_size, num_examples)
                    for celltype_idx in xrange(y.shape[1]):
                        batch_sequence = np.reshape(X_train[offset:end, :],
                                                    (self.batch_size, self.height, self.width, self.num_channels))
                        batch_labels = np.reshape(y_train[offset:end, celltype_idx], (self.batch_size, 1))
                        feed_dict = {self.tf_sequence: batch_sequence,
                                     self.tf_train_labels: batch_labels,
                                     self.keep_prob: 0.5}
                        _, r_loss = session.run([optimizer, loss], feed_dict=feed_dict)
                        '''
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        summary, _, r_loss = session.run([summary_op, optimizer, loss],
                                                         feed_dict=feed_dict,
                                                         options=run_options,
                                                         run_metadata=run_metadata)
                        if self.model_dir is not None:
                            summary_writer.add_summary(summary)
                            summary_writer.add_run_metadata(run_metadata, 'Epoch %d, offset %d' % (epoch, offset))
                        '''
                        losses.append(r_loss)
                t_loss = np.mean(np.array(losses))

                #loss_summary = tf.scalar_summary('Mean loss training', t_loss)
                #summary_writer.add_summary(loss_summary)

                # Validation
                accuracies = []
                losses = []
                num_examples = X_val.shape[0]
                prediction_op = tf.nn.sigmoid(self.logits)
                for offset in xrange(0, num_examples-self.batch_size, self.batch_size):
                    end = min(offset+self.batch_size, num_examples)
                    for celltype_idx in xrange(y.shape[1]):
                        batch_sequence = np.reshape(X_val[offset:end, :],
                                                    (self.batch_size, self.height, self.width, self.num_channels))
                        batch_labels = np.reshape(y_val[offset:end, celltype_idx], (self.batch_size, 1))
                        feed_dict = {self.tf_sequence: batch_sequence,
                                     self.tf_train_labels: batch_labels,
                                     self.keep_prob: 1}
                        prediction, valid_loss = session.run([prediction_op, loss], feed_dict=feed_dict)
                        accuracies.append(100.0*np.sum(np.abs(prediction-batch_labels) < 0.5)/batch_labels.size)
                        losses.append(valid_loss)
                v_acc = np.mean(np.array(accuracies))
                v_loss = np.mean(np.array(losses))

                early_score = self.early_stopping.update(v_loss)
                if early_score == 2:
                    saver.save(session, self.model_dir + 'conv.ckpt')
                    if self.verbose:
                        print (bcolors.OKCYAN+"%d\t%f\t%f\t%.2f%%\t\t%ds"+bcolors.ENDC) % \
                              (epoch, float(t_loss), float(v_loss), float(v_acc), int(time.time() - start_time))
                        #print bcolors.OKGREEN+"EPOCH %d\t\tTRAIN LOSS %f\t\tVALID LOSS %f\t\tACC %.2f%%\t\tTIME %ds"+bcolors.ENDC \
                        #      % (epoch, float(t_loss), float(v_loss), float(v_acc), int(time.time() - start_time))
                elif early_score == 1:
                    if self.verbose:
                        print "%d\t%f\t%f\t%.2f%%\t\t%ds" % \
                              (epoch, float(t_loss), float(v_loss), float(v_acc), int(time.time() - start_time))
                        #print "EPOCH %d\t\tTRAIN LOSS %f\t\tVALID LOSS %f\t\tACC %.2f%%\t\tTIME %ds" \
                        #      % (epoch, float(t_loss), float(v_loss), float(v_acc), int(time.time() - start_time))
                elif early_score == 0:
                    if self.verbose:
                        print "Early stopping triggered, exiting..."
            summary_writer.add_graph(session.graph)

    def predict(self, X):
        '''
        Run trained model
        :return: predictions
        '''
        prediction_op = tf.nn.sigmoid(self.logits)
        num_examples = X.shape[0]
        saver = tf.train.Saver()
        predictions = []
        with tf.Session() as session:
            saver.restore(session, self.model_dir+'conv.ckpt')
            for offset in xrange(0, num_examples, self.batch_size):
                end = min(offset + self.batch_size, num_examples)
                offset_ = offset - (self.batch_size-(end-offset))
                batch_sequence = np.reshape(X[offset_:end, :],
                                            (self.batch_size, self.height, self.width, self.num_channels))

                feed_dict = {self.tf_sequence: batch_sequence,
                             self.keep_prob: 1}
                prediction = session.run([prediction_op], feed_dict=feed_dict)
                prediction = prediction[0][offset-offset_:]
                predictions.extend(prediction)
        predictions = np.array(predictions).flatten()
        return predictions

    def get_output_for_dense_layer(self, X):
        return


