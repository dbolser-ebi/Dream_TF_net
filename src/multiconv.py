import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers.python.layers import *
import random
from datagen import *
from sklearn.cross_validation import KFold
import pdb


class configuration:
    SEQ = 1
    SEQ_SHAPE = 2
    SEQ_SHAPE_GENEXPR = 3
    SEQ_SHAPE_GENEXPR_ALLUSUAL = 4
    SEQ_SHAPE_SPECIFICUSUAL = 5
    SEQ_SHAPE_GENEXPR_ALLUSUAL_DNASE = 6
    SEQ_DNASE = 7
    SEQ_DNASE_SHAPE = 8
    SEQ_DNASE_SHAPE_ALLUSUAL = 9
    USUAL_DNASE = 10
    DNASE = 11


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
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def conv1D(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='VALID')


def max_pool_1xn(x, width):
    return tf.nn.max_pool(x, ksize=[1, 1, width, 1],
                          strides=[1, 1, width, 1], padding='VALID')


def calc_loss(logits, labels):
    logits = tf.reshape(logits, [-1])
    labels = tf.reshape(labels, [-1])
    index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))
    logits_known = tf.gather(logits, index)
    labels_known = tf.gather(labels, index)
    entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits_known, labels_known)

    return tf.reduce_mean(entropies)


def calc_loss_for_tf(logits, labels, tf_index):
    logits = logits[:, tf_index]
    labels = labels[:, tf_index]
    entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(entropies)


def calc_entropies(logits, labels, tf_index):
    '''
    logits = tf.reshape(logits, [-1])
    labels = tf.reshape(labels, [-1])
    comparison = tf.not_equal(labels, tf.constant(-1, dtype=tf.float32))
    index = tf.where(comparison)
    logits_known = tf.gather(logits, index)
    labels_known = tf.gather(labels, index)
    entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits_known, labels_known)
    means = tf.reduce_mean(entropies)
    '''
    logits = logits[:, tf_index]
    predictions = tf.nn.sigmoid(logits)
    return logits, labels, predictions


def calc_regression_loss(prediction, actual):
    mse = tf.reduce_mean(tf.square(tf.sub(prediction, actual)))
    return mse


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


class MultiConvNet:
    def __init__(self, model_dir, batch_size=256, num_channels=4, num_epochs=2,
                 sequence_width=200, num_outputs=1, eval_size=0.2, early_stopping=100,
                num_dnase_features=4, dropout_rate=.5,
                 config=1, verbose=True, name='convnet', segment='train', learning_rate=0.001):
        if config is None:
            config = 1
        self.learning_rate = learning_rate
        self.segment = segment
        self.name = name
        self.config = config
        self.num_dnase_features = num_dnase_features
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        self.height = 1
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.sequence_width = sequence_width
        self.eval_size = eval_size
        self.verbose = verbose
        self.tf_dnase_features = tf.placeholder(tf.float32, shape=(batch_size, num_dnase_features), name='dnase_features')
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          sequence_width, self.num_channels), name='sequences')
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_outputs), name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.trans_f_index = tf.placeholder(tf.int32, name='tf_index')
        self.early_stopping = EarlyStopping(max_stalls=early_stopping)
        self.dropout_rate = dropout_rate
        self.logits = self.get_model()
        self.datagen = DataGenerator()

    def get_model(self):
        with tf.variable_scope(self.name) as main_scope:
            with tf.variable_scope('DNASE') as scope:
                dnase_features = self.tf_dnase_features

            with tf.variable_scope('R_MOTIF') as scope:
                with tf.variable_scope('conv15_15') as convscope:
                    depth = 30
                    width = 15
                    rmotif_conv_kernel = weight_variable(shape=[self.height, width, self.num_channels, depth])

                    conv_biases = weight_variable(shape=[depth])
                    conv = conv1D(self.tf_sequence, rmotif_conv_kernel)
                    activation = tf.nn.bias_add(conv, conv_biases)

                with tf.variable_scope('pool35') as poolscope:
                    pool_width = 35
                    pool = tf.nn.relu(max_pool_1xn(activation, pool_width))

                with tf.variable_scope('fc100') as fcscope:
                    flattened = flatten(pool)
                    drop_rmotif = fully_connected(flattened, 100)

            with tf.variable_scope('MERGE') as scope:
                with tf.variable_scope('merge_drop') as mergescope:
                    if self.config == int(configuration.SEQ):
                        merged = tf.concat(1, [drop_rmotif])
                    elif self.config == int(configuration.SEQ_DNASE):
                        merged = tf.concat(1, [drop_rmotif, dnase_features])
                    elif self.config == int(configuration.DNASE):
                        merged = tf.concat(1, [dnase_features])

                with tf.variable_scope('fc') as fcscope:
                    activation = fully_connected(merged, 1000)
                    drop = tf.nn.dropout(activation, self.keep_prob)

                with tf.variable_scope('output') as outscope:
                    logits = fully_connected(drop, self.num_outputs, None)

        return logits

    def fit_all(self, celltypes):
        summary_writer = tf.train.SummaryWriter(self.model_dir + self.segment)
        try:
            with tf.variable_scope(self.name + '_opt') as scope:
                loss = calc_loss(self.logits, self.tf_train_labels)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(
                    loss)
                debug = calc_entropies(self.logits, self.tf_train_labels, self.trans_f_index)

            saver = tf.train.Saver(
                [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            with tf.Session() as session:
                tf.initialize_all_variables().run()

                if self.verbose:
                    print
                    print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

                ids = range(1000000)
                X = self.datagen.get_sequece_from_ids(ids, self.segment)
                num_examples = X.shape[0]

                # Training
                for epoch in xrange(1, self.num_epochs + 1):
                    train_losses = []
                    start_time = time.time()
                    comb_counter = 0

                    for celltype in celltypes:

                        print "Loading data for", len(ids), "examples"
                        da = np.load('../data/preprocess/DNASE_FEATURES_NORM/%s_%s_600.gz.npy'
                                     % (celltype, self.segment))[ids]
                        y = np.load('../data/preprocess/features/y_%s.npy' % celltype)[ids]
                        print 'Data for celltype', celltype, 'loaded.'

                        shuffle_idxs = np.arange(X.shape[0])
                        np.random.shuffle(shuffle_idxs)
                        X = X[shuffle_idxs]
                        y = y[shuffle_idxs]
                        da = da[shuffle_idxs]

                        comb_counter += 1

                        batch_train_losses = []
                        for offset in xrange(0, num_examples - self.batch_size, self.batch_size):
                            end = offset + self.batch_size
                            batch_sequence = np.reshape(X[offset:end, :, :],
                                                        (self.batch_size, self.height, self.sequence_width,
                                                         self.num_channels))
                            batch_labels = np.reshape(y[offset:end, :], (self.batch_size, self.num_outputs))
                            batch_dnase_features = np.reshape(da[offset:end, :self.num_dnase_features],
                                                              (self.batch_size, self.num_dnase_features))
                            feed_dict = {self.tf_sequence: batch_sequence,
                                         self.tf_train_labels: batch_labels,
                                         self.keep_prob: 1 - self.dropout_rate,
                                         self.tf_dnase_features: batch_dnase_features,
                                         }
                            _, r_loss = \
                                session.run([optimizer, loss], feed_dict=feed_dict)
                            batch_train_losses.append(r_loss)
                        print "Batch loss", np.mean(np.array(batch_train_losses))
                        train_losses.extend(batch_train_losses)

                    train_losses = np.array(train_losses)
                    t_loss = np.mean(train_losses)

                    early_score = self.early_stopping.update(t_loss)
                    if early_score == 2:
                        # Use the best model on validation
                        saver.save(session, self.model_dir + 'conv%s%s%d%d%d.ckpt'
                                   % (self.name, 'multiconv', self.config,
                                      self.num_dnase_features, self.sequence_width))
                        if self.verbose:
                            print (bcolors.OKCYAN + "%d\t%f\t%f\t\t%ds" + bcolors.ENDC) % \
                                  (epoch, float(t_loss), float(t_loss), int(time.time() - start_time))
                    elif early_score == 1:
                        if self.verbose:
                            print "%d\t%f\t%f\t\t%ds" % \
                                  (epoch, float(t_loss), float(t_loss), int(time.time() - start_time))
                    elif early_score == 0:
                        if self.verbose:
                            print "Early stopping triggered, exiting..."
                            break
                    summary_writer.add_graph(session.graph)

        except KeyboardInterrupt:
            pass

    def fit_tf(self, celltypes):
        summary_writer = tf.train.SummaryWriter(self.model_dir + self.segment)
        try:
            with tf.variable_scope(self.name+'_opt') as scope:
                loss = calc_loss_for_tf(self.logits, self.tf_train_labels, self.trans_f_index)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
                debug = calc_entropies(self.logits, self.tf_train_labels, self.trans_f_index)

            saver = tf.train.Saver([var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            with tf.Session() as session:
                tf.initialize_all_variables().run()

                if self.verbose:
                    print
                    print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

                #ids = list(self.datagen.get_bound_for_trans_f('CTCF'))
                #ids.extend(random.sample(xrange(51676736), len(ids)))
                ids = range(1000000)
                X = self.datagen.get_sequece_from_ids(ids, self.segment)
                num_examples = X.shape[0]
                trans_f_lookup = self.datagen.get_trans_f_lookup()

                # Training
                for epoch in xrange(1, self.num_epochs+1):
                    train_losses = []
                    start_time = time.time()
                    comb_counter = 0

                    for celltype in celltypes:

                        print "Loading data for", len(ids), "examples"
                        da = np.load('../data/preprocess/DNASE_FEATURES_NORM/%s_%s_600.gz.npy'
                                     % (celltype, self.segment))[ids]
                        y = np.load('../data/preprocess/features/y_%s.npy' % celltype)[ids]
                        print 'Data for celltype', celltype, 'loaded.'

                        shuffle_idxs = np.arange(X.shape[0])
                        np.random.shuffle(shuffle_idxs)
                        X = X[shuffle_idxs]
                        y = y[shuffle_idxs]
                        da = da[shuffle_idxs]

                        for trans_f in self.datagen.get_trans_fs():
                            if trans_f != 'CTCF':
                                continue
                            if celltype not in self.datagen.get_celltypes_for_trans_f(trans_f):
                                continue
                            comb_counter += 1
                            print "Training on transcription factor", trans_f, '/', comb_counter

                            trans_f_idx = trans_f_lookup[trans_f]

                            batch_train_losses = []
                            for offset in xrange(0, num_examples-self.batch_size, self.batch_size):
                                end = offset+self.batch_size
                                batch_sequence = np.reshape(X[offset:end, :, :],
                                                            (self.batch_size, self.height, self.sequence_width, self.num_channels))
                                batch_labels = np.reshape(y[offset:end], (self.batch_size, self.num_outputs))
                                batch_dnase_features = np.reshape(da[offset:end, :self.num_dnase_features],
                                                                  (self.batch_size, self.num_dnase_features))
                                feed_dict = {self.tf_sequence: batch_sequence,
                                             self.tf_train_labels: batch_labels,
                                             self.keep_prob: 1-self.dropout_rate,
                                             self.tf_dnase_features: batch_dnase_features,
                                             self.trans_f_index: trans_f_idx
                                             }
                                _, r_loss = \
                                    session.run([optimizer, loss], feed_dict=feed_dict)
                                batch_train_losses.append(r_loss)
                            print "Batch loss", np.mean(np.array(batch_train_losses))
                            train_losses.extend(batch_train_losses)

                    train_losses = np.array(train_losses)
                    t_loss = np.mean(train_losses)

                    early_score = self.early_stopping.update(t_loss)
                    if early_score == 2:
                        # Use the best model on validation
                        saver.save(session, self.model_dir + 'conv%s%s%d%d%d.ckpt'
                                   % (self.name, 'multiconv', self.config,
                                      self.num_dnase_features, self.sequence_width))
                        if self.verbose:
                            print (bcolors.OKCYAN+"%d\t%f\t%f\t\t%ds"+bcolors.ENDC) % \
                                  (epoch, float(t_loss), float(t_loss), int(time.time() - start_time))
                    elif early_score == 1:
                        if self.verbose:
                            print "%d\t%f\t%f\t\t%ds" % \
                                  (epoch, float(t_loss), float(t_loss), int(time.time() - start_time))
                    elif early_score == 0:
                        if self.verbose:
                            print "Early stopping triggered, exiting..."
                            break
                    summary_writer.add_graph(session.graph)

        except KeyboardInterrupt:
            pass

    def predict(self, celltype):
        '''
        Run trained model
        :return: predictions
        '''
        prediction_op = tf.nn.sigmoid(self.logits)

        if self.segment == 'train':
            num_test_indices = 51676736
        if self.segment == 'ladder':
            num_test_indices = 8843011
        if self.segment == 'test':
            num_test_indices = 60519747

        num_test_indices = 2702470

        stride = 1000000
        predictions = []
        for start in range(0, num_test_indices, stride):
            ids = range(start, min(start+stride, num_test_indices))
            X = self.datagen.get_sequece_from_ids(ids, self.segment)
            num_examples = X.shape[0]
            saver = tf.train.Saver(
                [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            # load dnase features
            dnase_features = np.load('../data/preprocess/DNASE_FEATURES_NORM/%s_%s_600.gz.npy'
                         % (celltype, self.segment))[ids]

            with tf.Session() as session:
                saver.restore(session, self.model_dir+'conv%s%s%d%d%d.ckpt'
                              % (self.name, 'multiconv', self.config,
                                 self.num_dnase_features, self.sequence_width))
                for offset in xrange(0, num_examples, self.batch_size):
                    end = min(offset + self.batch_size, num_examples)
                    offset_ = offset - (self.batch_size-(end-offset))
                    batch_sequence = np.reshape(X[offset_:end, :],
                                                (self.batch_size, self.height, self.sequence_width, self.num_channels))

                    batch_da_features = np.reshape(dnase_features[offset_:end, :self.num_dnase_features],
                                          (self.batch_size, self.num_dnase_features))
                    feed_dict = {self.tf_sequence: batch_sequence,
                                 self.keep_prob: 1,
                                 self.tf_dnase_features: batch_da_features,
                                 }
                    prediction = session.run([prediction_op], feed_dict=feed_dict)
                    prediction = prediction[0][offset-offset_:]
                    predictions.extend(prediction)
        predictions = np.array(predictions)
        return predictions
