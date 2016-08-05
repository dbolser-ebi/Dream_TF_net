import tensorflow as tf
import numpy as np
from sklearn.cross_validation import StratifiedKFold


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


def calc_val_acc(logits, labels):
    predictions = tf.nn.sigmoid(logits)
    return


class ConvNet:
    def __init__(self, model_dir, batch_size=128, num_channels=4, num_epochs=2,
                 width=200, num_outputs=1, eval_size=0.2, verbose=True):
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
        self.dropout_rate = tf.placeholder(tf.float32)

        self.logits = self.get_model()
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
            drop = tf.nn.dropout(fc1, self.dropout_rate)

        # Output
        with tf.variable_scope('softmax') as scope:
            fc_kernel2 = weight_variable([100, 1])
            fc_bias2 = bias_variable([1])
            logits = tf.matmul(drop, fc_kernel2) + fc_bias2

        return logits

    def fit(self, X, y):
        summary_writer = tf.train.SummaryWriter(self.model_dir + 'train')

        loss = calc_loss(self.logits, self.tf_train_labels)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        num_examples = X.shape[0]
        saver = tf.train.Saver()

        y_ = np.max(y, axis=1)
        kf = StratifiedKFold(y_, round(1. / self.eval_size))
        train_indices, valid_indices = next(iter(kf))
        X_train = X[train_indices]
        X_val = X[valid_indices]
        y_train = y[train_indices]
        y_val = X[valid_indices]

        with tf.Session() as session:
            tf.initialize_all_variables().run()

            # train model
            for epoch in xrange(1, self.num_epochs+1):
                losses = []
                # run minibatches
                for offset in xrange(0, num_examples-self.batch_size, self.batch_size):
                    end = min(offset+self.batch_size, num_examples)
                    for celltype_idx in xrange(y.shape[1]):
                        batch_sequence = np.reshape(X[offset:end, :],
                                                    (self.batch_size, self.height, self.width, self.num_channels))
                        batch_labels = np.reshape(y[offset:end, celltype_idx], (self.batch_size, 1))
                        feed_dict = {self.tf_sequence: batch_sequence,
                                     self.tf_train_labels: batch_labels,
                                     self.dropout_rate: 0.5}
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
                t_loss = np.sum(np.array(losses))

                #loss_summary = tf.scalar_summary('Mean loss training', t_loss)
                #summary_writer.add_summary(loss_summary)

                if self.verbose:
                    print epoch, t_loss

            saver.save(session, self.model_dir+'conv.ckpt')
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
                             self.dropout_rate: 1}
                prediction = session.run([prediction_op], feed_dict=feed_dict)
                print len(prediction[0]),
                prediction = prediction[0][offset-offset_:]
                print offset, offset_, end, len(prediction[0])
                predictions.extend(prediction)
        predictions = np.array(predictions).flatten()
        return predictions

    def get_output_for_layer(self, sequence):
        return


