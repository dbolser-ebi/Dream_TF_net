import tensorflow as tf
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
import time
from tensorflow.contrib.layers.python.layers import *
import xgboost as xgb
import pdb
from datagen import *


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


def calc_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))


def calc_loss_seperate(logits, labels, ratio):
    entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    labels_complement = tf.constant(1.0, dtype=tf.float32) - labels
    entropy_bound = tf.reduce_sum(tf.mul(labels, entropies))
    entropy_unbound = tf.reduce_sum(tf.mul(labels_complement, entropies))
    num_bound = tf.reduce_sum(labels)
    num_unbound = tf.reduce_sum(labels_complement)
    loss_bound = tf.mul(ratio, tf.cond(tf.equal(num_bound, tf.constant(0.0)), lambda: tf.constant(0.0),
                         lambda: tf.div(entropy_bound, num_unbound)))
    loss_unbound = tf.div(entropy_unbound, num_unbound)
    return tf.add(loss_bound, loss_unbound)


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


class ConvNet:
    def __init__(self, model_dir, batch_size=256, num_channels=4, num_epochs=2,
                 sequence_width=200, num_outputs=1, eval_size=0.2, early_stopping=100,
                 num_gen_expr_features=57820, num_dnase_features=4, num_shape_features=4, dropout_rate=.5,
                 config=1, verbose=True, transcription_factor='RFX5', regression=False,
                 name='convnet', num_chunks=10, id='ID', debug=False):
        if config is None:
            config = 1
        np.set_printoptions(suppress=True)
        np.random.seed(12345)

        self.id = id
        self.debug = debug
        self.num_chunks = min(num_chunks, 52)
        self.name = name
        self.regression = regression
        self.datagen = DataGenerator()
        self.config = config
        self.num_dnase_features = num_dnase_features
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        self.num_shape_features = num_shape_features
        self.height = 1
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.bin_size = sequence_width
        self.eval_size = eval_size
        self.verbose = verbose
        self.num_genexpr_features = num_gen_expr_features
        self.tf_ratio = tf.placeholder(tf.float32, name='bound_ratio')
        self.tf_gene_expression = tf.placeholder(tf.float32, shape=(1, num_gen_expr_features), name='tpm_values')
        self.tf_dnase_features = tf.placeholder(tf.float32, shape=(batch_size, self.num_dnase_features), name='dnase_features')
        self.tf_shape = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                                          sequence_width, num_shape_features), name='shapes')
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          sequence_width, self.num_channels), name='sequences')
        self.tf_boley_scores = tf.placeholder(tf.float32, (self.batch_size, 7), name='boley_scores')
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs), name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.early_stopping = EarlyStopping(max_stalls=early_stopping)
        self.dropout_rate = dropout_rate
        self.transcription_factor = transcription_factor
        self.tf_features = tf.placeholder(tf.float32, shape=(batch_size, self.height, sequence_width, self.num_channels+1))
        self.original_pssm = None
        self.flatted_activations, self.logits = self.get_combined_model()

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def get_combined_model(self):
        with tf.variable_scope(self.name) as main_scope:
            with tf.variable_scope('R_MOTIF') as scope:

                with tf.variable_scope('conv1') as convscope:
                    kernel_width = 30
                    num_filters = 10
                    conv1 = convolution2d(self.tf_sequence, num_filters, [self.height, kernel_width], activation_fn=None)

                with tf.variable_scope('pool') as poolscope:
                    pool_width = 35
                    pool = tf.nn.relu(max_pool2d(conv1, (self.height, pool_width), stride=pool_width,padding='SAME'))

                with tf.variable_scope('fc100') as fcscope:
                    flattened = flatten(pool)
                    #flattened = tf.concat(1, [self.tf_boley_scores])
                    drop_rmotif = fully_connected(flattened, 100)
                    drop_rmotif = tf.nn.dropout(drop_rmotif, self.keep_prob)

                with tf.variable_scope('output') as outscope:
                    logits = fully_connected(drop_rmotif, 1, None)

        return flattened, logits

    def fit_combined(self, celltypes_train):
        summary_writer = tf.train.SummaryWriter(self.model_dir + 'train')
        try:
            with tf.variable_scope(self.name+'_opt') as scope:
                loss = calc_loss_seperate(self.logits, self.tf_train_labels, self.tf_ratio)  #calc_loss(self.logits, self.tf_train_labels)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(loss)

            saver = tf.train.Saver([var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            with tf.Session() as session:
                tf.initialize_all_variables().run()

                if self.verbose:
                    print
                    print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

                batch_features = np.zeros((self.batch_size, self.height, self.bin_size, self.num_channels+1))

                # Training

                trans_f_idx = self.datagen.get_trans_f_lookup()[self.transcription_factor]

                boley_scores = self.datagen.get_boley_scores('train', 200, 'TAF1')

                for epoch in xrange(1, self.num_epochs+1):
                    train_losses = []
                    start_time = time.time()

                    for c_idx, celltype in enumerate(celltypes_train):
                        y = self.datagen.get_y(celltype)
                        dnase_features = self.datagen.get_dnase_features('train', celltype, 200)
                        print 'Data for celltype', celltype, 'loaded.'

                        for chunk_id in range(1, self.num_chunks+1):
                            ids = range((chunk_id - 1) * 1000000, min(chunk_id * 1000000, self.datagen.train_length))

                            X = self.datagen.get_sequece_from_ids(ids, 'train', self.bin_size)
                            num_examples = X.shape[0]
                            y_chunk = y[ids]
                            dnase_chunk = dnase_features[ids]

                            boley_scores_chunk = boley_scores[ids]

                            # Batch stratification and shuffling

                            bound_idxs = np.where(y_chunk[:, trans_f_idx] == 1)[0]
                            unbound_idxs = np.where(y_chunk[:, trans_f_idx] == 0)[0]

                            np.random.shuffle(bound_idxs)
                            np.random.shuffle(unbound_idxs)
                            shuffle_idxs = np.zeros((y_chunk.shape[0]), dtype=np.int32)
                            chunk_ratio = unbound_idxs.shape[0]/bound_idxs.shape[0]
                            shuffle_it = 0
                            for shuffle_it in range(bound_idxs.shape[0]):
                                offset = shuffle_it*(chunk_ratio+1)
                                shuffle_idxs[offset] = bound_idxs[shuffle_it]
                                shuffle_idxs[offset+1:offset+chunk_ratio+1] = unbound_idxs[shuffle_it*chunk_ratio:(shuffle_it+1)*chunk_ratio]
                            shuffle_it += 1
                            shuffle_idxs[shuffle_it*(chunk_ratio+1):] = unbound_idxs[shuffle_it*chunk_ratio:]

                            X = X[shuffle_idxs]
                            y_chunk = y_chunk[shuffle_idxs]
                            dnase_chunk = dnase_chunk[shuffle_idxs]

                            print "Loaded %d examples for batch %d" % (X.shape[0], chunk_id)

                            batch_train_losses = []
                            for offset in xrange(0, num_examples - self.batch_size, self.batch_size):
                                end = offset + self.batch_size
                                batch_sequence = np.reshape(X[offset:end, :, :],
                                                            (self.batch_size, self.height, self.bin_size,
                                                             self.num_channels))

                                batch_features[:, :, :, :self.num_channels] = batch_sequence

                                if self.config == 1:
                                    batch_features[:, :, :, self.num_channels] = \
                                        np.zeros((self.batch_size, self.height, self.bin_size), dtype=np.float32)
                                else:
                                    batch_features[:, :, :, self.num_channels] = \
                                        np.reshape(dnase_chunk[offset:end, :],
                                                   (self.batch_size, self.height, self.bin_size))
                                batch_labels = np.reshape(y_chunk[offset:end, trans_f_idx], (self.batch_size, self.num_outputs))
                                feed_dict = {
                                            self.tf_boley_scores: boley_scores_chunk[offset:end],
                                            self.tf_sequence: batch_sequence,
                                            self.tf_dnase_features: dnase_chunk[offset:end, :],
                                            self.tf_train_labels: batch_labels,
                                            self.keep_prob: 1 - self.dropout_rate,
                                            self.tf_ratio: chunk_ratio,
                                             }

                                _, r_loss = \
                                    session.run([optimizer, loss], feed_dict=feed_dict)

                                batch_train_losses.append(r_loss)

                            print "Batch loss", np.mean(np.array(batch_train_losses))
                            train_losses.extend(batch_train_losses)

                            if self.debug:
                                vars = [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if
                                 var.op.name.startswith(self.name)]
                                for bound_switch in ['bound', 'unbound']:
                                    b = 1 if bound_switch == 'bound' else 0
                                    bound_idxs = np.where(y_chunk[:, trans_f_idx] == b)[0]
                                    np.random.shuffle(bound_idxs)
                                    num_examples = bound_idxs.shape[0]

                                    mean_activations = []
                                    dnase_profile = np.sum(dnase_chunk[bound_idxs], axis=0)

                                    for offset in xrange(0, num_examples - self.batch_size, self.batch_size):
                                        end = offset + self.batch_size
                                        batch_sequence = np.reshape(X[bound_idxs[offset:end], :, :],
                                                                    (self.batch_size, self.height, self.bin_size,
                                                                     self.num_channels))
                                        batch_features[:, :, :, :self.num_channels] = batch_sequence

                                        if self.config == 1:
                                            batch_features[:, :, :, self.num_channels] = \
                                                np.zeros((self.batch_size, self.height, self.bin_size),
                                                         dtype=np.float32)
                                        else:
                                            batch_features[:, :, :, self.num_channels] = \
                                                np.reshape(dnase_chunk[bound_idxs[offset:end], :],
                                                           (self.batch_size, self.height, self.bin_size))

                                        batch_labels = np.reshape(y_chunk[bound_idxs[offset:end], trans_f_idx],
                                                                  (self.batch_size, self.num_outputs))
                                        feed_dict = {
                                            self.tf_boley_scores: boley_scores_chunk[offset:end],
                                            self.tf_sequence: batch_sequence,
                                            self.tf_dnase_features: dnase_chunk[bound_idxs[offset:end]],
                                            self.tf_train_labels: batch_labels,
                                            self.keep_prob: 1,
                                        }
                                        activations = session.run(self.flatted_activations, feed_dict=feed_dict)

                                        mean_activations.append(np.mean(activations))
                                    mean_activation = np.mean(np.array(mean_activations))
                                    dnase_profile /= bound_idxs.shape[0]
                                    pdb.set_trace()

                    train_losses = np.array(train_losses)
                    t_loss = np.mean(train_losses)

                    early_score = self.early_stopping.update(t_loss)
                    if early_score == 2:
                        # Use the best model on validation
                        saver.save(session, self.model_dir + 'conv%s.ckpt'
                                   % self.id)
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

    def predict_combined(self, X, dnase_features, boley_scores):
        '''
                Run trained model
                :return: predictions
                '''
        prediction_op = tf.nn.sigmoid(self.logits)
        num_examples = X.shape[0]
        saver = tf.train.Saver(
            [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])
        predictions = []
        batch_features = np.zeros((self.batch_size, self.height, self.bin_size, self.num_channels+1))
        with tf.Session() as session:
            saver.restore(session, self.model_dir + 'conv%s.ckpt'
                                   % self.id)
            for offset in xrange(0, num_examples, self.batch_size):
                end = min(offset + self.batch_size, num_examples)
                offset_ = offset - (self.batch_size - (end - offset))
                batch_sequence = np.reshape(X[offset_:end, :], (self.batch_size, self.height, self.bin_size, self.num_channels))
                batch_features[:, :, :, :self.num_channels] = \
                    np.reshape(X[offset_:end, :], (self.batch_size, self.height, self.bin_size, self.num_channels))
                if self.config == 1:
                    batch_features[:, :, :, self.num_channels] = \
                        np.zeros((self.batch_size, self.height, self.bin_size), dtype=np.float32)
                else:
                    batch_features[:, :, :, self.num_channels] = \
                        np.reshape(dnase_features[offset_:end, :], (self.batch_size, self.height,
                                                                    self.bin_size))
                feed_dict = {
                            self.tf_boley_scores: boley_scores[offset_:end],
                            self.tf_dnase_features: dnase_features[offset_:end],
                            self.tf_sequence: batch_sequence,
                             self.keep_prob: 1,
                             }
                prediction = session.run([prediction_op], feed_dict=feed_dict)
                prediction = prediction[0][offset - offset_:]
                predictions.extend(prediction)
        predictions = np.array(predictions).flatten()
        return predictions


class XGBoost:
    def __init__(self, batch_size=256, sequence_width=200, config=1, transcription_factor='CTCF',
                    num_channels=4, height=1, datapath='../data/'):
        self.height = height
        self.num_channels = num_channels
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          sequence_width, self.num_channels), name='sequences')
        self.batch_size = batch_size
        self.bin_size = sequence_width
        self.datagen = DataGenerator()
        self.config = config
        self.transcription_factor = transcription_factor
        self.activations, self.summary_op = self.get_activations()

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def fit(self, X, y, S=None, gene_expression=None, da=None, y_quant=None):
        activations = []
        number_of_batches = X.shape[0] / self.batch_size
        offset = X.shape[0] % self.batch_size
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(number_of_batches):
                batch_activations = sess.run(self.activations, feed_dict={self.tf_sequence:
                                                                              X[self.batch_size*i:
                                                                              self.batch_size*i+self.batch_size].
                                             reshape(self.batch_size, self.height, self.bin_size,
                                                     self.num_channels)}
                                             )
                batch_activations = np.concatenate(batch_activations, axis=1)
                activations.append(batch_activations)
            if offset != 0:
                zero_array = np.zeros([self.batch_size - offset, self.bin_size, self.num_channels])
                final_chunk = np.concatenate([X[self.batch_size*(i+1):], zero_array])
                batch_activations = sess.run(self.activations,
                                             feed_dict={self.tf_sequence:final_chunk.reshape(
                                                 self.batch_size,
                                                 self.height,
                                                 self.bin_size,
                                                 self.num_channels)})
                batch_activations = np.concatenate(batch_activations, axis=1)
                activations.append(batch_activations)

        activations = np.concatenate(activations)
        activations = activations[:offset-self.batch_size]
        da_features = np.concatenate([da[:, :, i] for i in range(da.shape[2])])
        full_activations = np.concatenate(([activations] * da.shape[2]))
        full_features = np.concatenate([full_activations, da_features], axis=1)
        full_y = np.concatenate([y[:, i] for i in range(y.shape[1])])


        dtrain = xgb.DMatrix(full_features, full_y)
        param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        plst = param.items()
        num_round = 10
        bst = xgb.train(plst, dtrain, num_round)
        self.model = bst
        print 'XGBoost trained'

    def predict(self, X, S=None, gene_expression=None, da=None):
        '''
        run trained model
        :return: predictions
        '''

        activations = []
        number_of_batches = X.shape[0] / self.batch_size
        offset = X.shape[0] % self.batch_size
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(number_of_batches):
                batch_activations = sess.run(self.activations,
                                             feed_dict={
                                                 self.tf_sequence:
                                                     X[self.batch_size*i:
                                                       self.batch_size*i+self.batch_size].reshape(
                                                       self.batch_size, self.height,
                                                       self.bin_size, self.num_channels)}
                                             )
                batch_activations = np.concatenate(batch_activations, axis=1)
                activations.append(batch_activations)
            if offset != 0:
                zero_array = np.zeros([self.batch_size - offset, self.bin_size, self.num_channels])
                final_chunk = np.concatenate([X[self.batch_size*(i+1):], zero_array])
                batch_activations = sess.run(self.activations,
                                             feed_dict={self.tf_sequence:final_chunk.reshape(self.batch_size,
                                                                                             self.height,
                                                                                             self.bin_size,
                                                                                             self.num_channels)})
                batch_activations = np.concatenate(batch_activations, axis=1)
                activations.append(batch_activations)

        activations = np.concatenate(activations)
        activations = activations[:offset-self.batch_size]
        da_features = da.reshape(X.shape[0], -1)
        features = np.concatenate([activations, da_features], axis=1)
        d_test = xgb.DMatrix(features)
        return self.model.predict(d_test)

    def get_activations(self):
        # with tf.variable_scope('DNASE') as scope:
            # dnase_features = self.tf_dnase_features

        with tf.variable_scope('USUAL_SUSPECTS') as scope:
            activations = []

            def get_activations_for_tf(transcription_factor):
                result = []
                motifs = self.datagen.get_motifs_h(transcription_factor)
                if len(motifs) > 0:
                    with tf.variable_scope(transcription_factor) as tfscope:
                        for idx, pssm in enumerate(motifs):
                            usual_conv_kernel = \
                                tf.get_variable('motif_%d' % idx,
                                                shape=(1, pssm.shape[0], pssm.shape[1], 1), dtype=tf.float32,
                                                initializer=tf.constant_initializer(pssm))
                            depth = 1
                            filter_width = pssm.shape[0]
                            conv_biases = tf.zeros(shape=[depth])
                            stride = 1
                            conv = conv1D(self.tf_sequence, usual_conv_kernel, strides=[1, 1, stride, 1])
                            num_nodes = (self.bin_size - filter_width) / stride + 1
                            denominator = 4
                            for div in range(4, 10):
                                if num_nodes % div == 0:
                                    denominator = div
                                    break
                            activation = tf.nn.bias_add(conv, conv_biases)
                            pooled = tf.nn.relu(max_pool_1xn(activation, num_nodes / denominator))
                            result.append(flatten(pooled))
                return result

            merged_summary = None

            activations.extend(get_activations_for_tf(self.transcription_factor))

        return activations, merged_summary

