import tensorflow as tf
import time
from tensorflow.contrib.layers.python.layers import *
from datagen import *
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


class EarlyStopping:
    def __init__(self, max_stalls):
        self.best_loss = 50000
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
                 config=1, verbose=True, name='convnet', segment='train', learning_rate=0.001, seperate_cost=False,
                 debug=False, id="ID", num_chunks=10):
        if config is None:
            config = 1
        self.id = id
        self.debug = debug
        self.seperate_cost = seperate_cost
        self.learning_rate = learning_rate
        self.num_chunks = num_chunks
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
        self.bin_size = sequence_width
        self.eval_size = eval_size
        self.verbose = verbose
        self.tf_features = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, sequence_width,
                                                              self.num_channels+1), name='features')
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          sequence_width, self.num_channels), name='sequences')
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_outputs), name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.trans_f_index = tf.placeholder(tf.int32, name='tf_index')
        self.early_stopping = EarlyStopping(max_stalls=early_stopping)
        self.dropout_rate = dropout_rate
        self.tf_ratio = tf.placeholder(tf.float32, name='ratio_bound_unbound')
        self.logits = self.get_model()
        self.datagen = DataGenerator()

    def set_segment(self, segment):
        self.segment = segment

    def get_model(self):
        with tf.variable_scope(self.name) as main_scope:
            with tf.variable_scope('R_MOTIF') as scope:
                with tf.variable_scope('conv1') as convscope:
                    kernel_width = 16
                    num_filters = 10
                    conv1 = convolution2d(self.tf_features, num_filters, [self.height, kernel_width], activation_fn=None)

                with tf.variable_scope('pool') as poolscope:
                    pool_width = 35
                    pool = tf.nn.relu(max_pool2d(conv1, (self.height, pool_width), stride=pool_width))

                with tf.variable_scope('fc100') as fcscope:
                    flattened = flatten(pool)
                    drop_rmotif = fully_connected(flattened, 100)
                    drop_rmotif = tf.nn.dropout(drop_rmotif, self.keep_prob)

                with tf.variable_scope('output') as outscope:
                    logits = fully_connected(drop_rmotif, self.num_outputs, None)
        return logits

    def get_loss(self, labels, logits):
        logits = tf.reshape(logits, [-1])
        labels = tf.reshape(labels, [-1])
        index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))
        logits_known = tf.gather(logits, index)
        labels_known = tf.gather(labels, index)
        entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits_known, labels_known)
        return tf.reduce_mean(entropies)

    def calc_loss_seperate(self, logits, labels, ratio):
        logits = tf.reshape(logits, [-1])
        labels = tf.reshape(labels, [-1])
        index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))
        logits = tf.gather(logits, index)
        labels = tf.gather(labels, index)
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

    def get_debug(self, labels, logits):
        predictions = tf.nn.sigmoid(logits)
        return labels, logits, predictions

    def fit(self, celltypes):
        summary_writer = tf.train.SummaryWriter(self.model_dir + self.segment)
        try:
            with tf.variable_scope(self.name + '_opt') as scope:
                loss = self.calc_loss_seperate(self.logits, self.tf_train_labels, self.tf_ratio)#self.get_loss(self.tf_train_labels, self.logits)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                   beta2=0.999).minimize(loss)

            saver = tf.train.Saver(
                [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            with tf.Session() as session:
                tf.initialize_all_variables().run()

                if self.verbose:
                    print
                    print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

                # Training
                for epoch in xrange(1, self.num_epochs + 1):
                    train_losses = []
                    start_time = time.time()
                    batch_features = np.zeros((self.batch_size, self.height, self.bin_size, self.num_channels+1), dtype=np.float32)
                    for chunk_id in range(1, self.num_chunks+1):
                        ids = range((chunk_id - 1) * 1000000, min(chunk_id * 1000000, self.datagen.train_length))
                        X = self.datagen.get_sequece_from_ids(ids, self.segment, self.bin_size)
                        num_examples = X.shape[0]

                        for celltype in celltypes:

                            print "Loading data for", len(ids), "examples"
                            y = np.load('../data/preprocess/features/y_%s.npy' % celltype)[ids]
                            dnase_features = self.datagen.get_dnase_features_from_ids(ids,
                                                                                      self.segment,
                                                                                      celltype,
                                                                                      self.bin_size)
                            print 'Data for celltype', celltype, 'loaded.'

                            batch_train_losses = []
                            for offset in xrange(0, num_examples - self.batch_size, self.batch_size):
                                end = offset + self.batch_size
                                batch_sequence = np.reshape(X[offset:end, :, :],
                                                            (self.batch_size, self.height, self.bin_size,
                                                             self.num_channels))
                                batch_features[:, :, :, :self.num_channels] = batch_sequence
                                batch_features[:, :, :, self.num_channels] = \
                                    np.reshape(dnase_features[offset:end, :],
                                               (self.batch_size, self.height, self.bin_size))

                                batch_labels = np.reshape(y[offset:end, :], (self.batch_size, self.num_outputs))
                                feed_dict = {self.tf_sequence: batch_sequence,
                                             self.tf_features: batch_features,
                                             self.tf_train_labels: batch_labels,
                                             self.keep_prob: 1 - self.dropout_rate,
                                             self.tf_ratio: 100,
                                             }

                                _, r_loss = \
                                    session.run([optimizer, loss], feed_dict=feed_dict)

                                batch_train_losses.append(r_loss)

                            print "Batch loss", np.mean(np.array(batch_train_losses))
                            train_losses.extend(batch_train_losses)

                            if self.debug:
                                # DEBUG ##############
                                bounds = []
                                unbounds = []

                                for offset in xrange(0, num_examples - self.batch_size, self.batch_size):
                                    if np.random.rand() > .9:
                                        end = offset + self.batch_size
                                        batch_sequence = np.reshape(X[offset:end, :, :],
                                                                    (self.batch_size, self.height, self.bin_size,
                                                                     self.num_channels))
                                        batch_features[:, :, :, :self.num_channels] = batch_sequence
                                        batch_features[:, :, :, self.num_channels] = \
                                            np.reshape(dnase_features[offset:end, :],
                                                       (self.batch_size, self.height, self.bin_size))

                                        batch_labels = np.reshape(y[offset:end, :], (self.batch_size, self.num_outputs))
                                        feed_dict = {self.tf_sequence: batch_sequence,
                                                     self.tf_features: batch_features,
                                                     self.tf_train_labels: batch_labels,
                                                     self.keep_prob: 1,
                                                     }
                                        labels, logits, predictions = \
                                            session.run(self.get_debug(self.tf_train_labels, self.logits), feed_dict=feed_dict)
                                        ind_bound = np.where(labels == 1)[0]
                                        ind_unbound = np.where(labels == 0)[0]
                                        bounds.extend(predictions[ind_bound].tolist())
                                        unbounds.extend(predictions[ind_unbound].tolist())
                                print 'num bound', len(bounds)
                                print "bound median", np.percentile(np.array(bounds), 50)
                                print "unbound median", np.percentile(np.array(unbounds), 50)
                                ########

                    train_losses = np.array(train_losses)
                    t_loss = np.mean(train_losses)

                    early_score = self.early_stopping.update(t_loss)
                    if early_score == 2:
                        # Use the best model on validation
                        saver.save(session, self.model_dir + 'multiconv%s.ckpt'
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

    def predict(self, celltype, validation=False):
        '''
        Run trained model
        :return: predictions
        '''
        dnase_features_total = self.datagen.get_dnase_features(self.segment, celltype, self.bin_size)
        prediction_op = tf.nn.sigmoid(self.logits)

        if self.segment == 'train':
            num_test_indices = 51676736
        if self.segment == 'ladder':
            num_test_indices = 8843011
        if self.segment == 'test':
            num_test_indices = 60519747
        if validation:
            num_test_indices = 2702470

        stride = 1000000
        predictions = []
        batch_features = np.zeros((self.batch_size, self.height, self.bin_size, self.num_channels+1))
        for start in range(0, num_test_indices, stride):
            ids = range(start, min(start+stride, num_test_indices))
            X = self.datagen.get_sequece_from_ids(ids, self.segment, self.bin_size)
            dnase_features = dnase_features_total[ids]
            num_examples = X.shape[0]
            saver = tf.train.Saver(
                [var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if var.op.name.startswith(self.name)])

            with tf.Session() as session:
                saver.restore(session, self.model_dir+'multiconv%s.ckpt'
                              % self.id)
                for offset in xrange(0, num_examples, self.batch_size):
                    end = min(offset + self.batch_size, num_examples)
                    offset_ = offset - (self.batch_size-(end-offset))
                    batch_sequence = np.reshape(X[offset_:end, :],(self.batch_size, self.height, self.bin_size, self.num_channels))
                    batch_features[:, :, :, :self.num_channels] = batch_sequence
                    batch_features[:, :, :, self.num_channels] = \
                        np.reshape(dnase_features[offset_:end, :],
                                   (self.batch_size, self.height, self.bin_size))

                    feed_dict = {self.tf_sequence: batch_sequence,
                                 self.tf_features: batch_features,
                                 self.keep_prob: 1
                                 }
                    prediction = session.run([prediction_op], feed_dict=feed_dict)
                    prediction = prediction[0][offset-offset_:]
                    predictions.extend(prediction)
        predictions = np.array(predictions)
        return predictions
