import tensorflow as tf
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import time
from datareader import DataReader
from sklearn.ensemble import RandomForestClassifier

def get_motifs_h(transcription_factor):
    motifs = []
    # Try Jaspar
    JASPAR_dir = '../data/preprocess/JASPAR/'
    for f in os.listdir(JASPAR_dir):
        if transcription_factor.upper() in f.upper():
            # print "motif found in JASPAR"
            motifs.append(np.loadtxt(os.path.join(JASPAR_dir, f), dtype=np.float32, unpack=True))

    # Try SELEX
    SELEX_dir = '../data/preprocess/SELEX_PWMs_for_Ensembl_1511_representatives/'
    for f in os.listdir(SELEX_dir):
        if f.upper().startswith(transcription_factor.upper()):
            # print "motif found in SELEX"
            motifs.append(np.loadtxt(os.path.join(SELEX_dir, f), dtype=np.float32, unpack=True))

    return motifs


def calc_pssm(pfm, pseudocounts=0.001):
    pfm += pseudocounts
    norm_pwm = pfm / pfm.sum(axis=1)[:, np.newaxis]
    return np.log2(norm_pwm / 0.25)



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


def get_X(self, model, num_instances):
    if isinstance(model, ConvNet):
        X = np.zeros((num_instances, 200, 4), dtype=np.float32)
    elif isinstance(model, RandomForestClassifier):
        num_sequence_features = self.datareader.get_num_sequence_features(tf)
        X = np.zeros((num_instances, num_sequence_features), dtype=np.float32)
    else:
        X = np.zeros((num_instances, 4, 200), dtype=np.float32)
    return X


def get_S(self, model, num_instances):
    if isinstance(model, ConvNet):
        S = np.zeros((num_instances, 200, 1), dtype=np.float32)
    else:
        S = np.zeros((num_instances, 1, 200), dtype=np.float32)
    return S


def get_y_train(self, model, num_instances, num_celltypes):
    if isinstance(model, ConvNet):
        y_train = np.zeros((num_instances, num_celltypes), dtype=np.float16)
    elif isinstance(model, RandomForestClassifier):
        y_train = np.zeros((num_instances,), dtype=np.int32)
    else:
        y_train = np.zeros((num_instances,), dtype=np.int32)
    return y_train


def get_y_test(self, model, num_instances):
    if isinstance(model, ConvNet):
        y_test = np.zeros((num_instances,), dtype=np.float16)
    elif isinstance(model, RandomForestClassifier):
        y_test = np.zeros((num_instances,), dtype=np.int32)
    else:
        y_test = np.zeros((num_instances,), dtype=np.int32)
    return y_test


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
                 width=200, num_outputs=1, eval_size=0.2, early_stopping=100,
                 num_gen_expr_features=57820, num_shape_features=1, dropout_rate=.5, verbose=True):
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        self.num_shape_features = num_shape_features
        self.height = 1
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.width = width
        self.eval_size = eval_size
        self.verbose = verbose
        self.num_genexpr_features = num_gen_expr_features
        self.tf_gene_expression = tf.placeholder(tf.float32, shape=(1, num_gen_expr_features), name='tpm_values')
        self.tf_dnase_accesible = tf.placeholder(tf.float32, shape=(batch_size, 1))
        self.tf_shape = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                                          width, num_shape_features), name='shapes')
        self.tf_sequence = tf.placeholder(tf.float32, shape=(batch_size, self.height,
                                          width, self.num_channels), name='sequences')
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs), name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.logits, self.summary_op = self.get_model()
        self.early_stopping = EarlyStopping(max_stalls=early_stopping)
        self.dropout_rate = dropout_rate

    def get_model(self):

        with tf.variable_scope('R_MOTIF') as scope:
            with tf.variable_scope('conv15_15') as convscope:
                depth = 15
                width = 15
                conv_kernel = weight_variable(shape=[self.height, width, self.num_channels, depth])
                conv_biases = weight_variable(shape=[depth])
                conv = conv1D(self.tf_sequence, conv_kernel)
                activation = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

            with tf.variable_scope('pool35') as poolscope:
                pool_width = 35
                pool = max_pool_1xn(activation, pool_width)

            with tf.variable_scope('fc100') as fcscope:
                fc_kernel = weight_variable([5 * depth, 100])
                fc_bias = bias_variable([100])
                flattened = tf.reshape(pool, [-1, 5 * depth])
                activation = tf.nn.relu(tf.matmul(flattened, fc_kernel) + fc_bias)

            with tf.variable_scope('dropout') as dropscope:
                drop_rmotif = tf.nn.dropout(activation, self.keep_prob)

            with tf.variable_scope('output') as outscope:
                fc_kernel = weight_variable([100, 1])
                fc_bias = bias_variable([1])
                logits = tf.matmul(drop_rmotif, fc_kernel) + fc_bias


        # TENSORBOARD SUMMARY INFO
        conv_kernel_h = tf.histogram_summary('convkernel histogram', conv_kernel)
        merged_summary = tf.merge_all_summaries()

        return logits, merged_summary

    def fit(self, X, y, S=None, gene_expression=None):
        summary_writer = tf.train.SummaryWriter(self.model_dir + 'train')

        loss = calc_loss(self.logits, self.tf_train_labels)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        saver = tf.train.Saver()

        y_ = np.max(y, axis=1)
        kf = StratifiedKFold(y_, round(1. / self.eval_size))
        train_indices, valid_indices = next(iter(kf))

        X_train = X[train_indices]
        X_val = X[valid_indices]

        S_train = S[train_indices]
        S_val = S[valid_indices]

        y_train = y[train_indices]
        y_val = y[valid_indices]

        print 'Train size', X_train.shape[0], 'Validation size', X_val.shape[0]

        with tf.Session() as session:
            tf.initialize_all_variables().run()

            if self.verbose:
                print
                print "EPOCH\tTRAIN LOSS\tVALID LOSS\tVALID ACCURACY\tTIME"

            # Training
            for epoch in xrange(1, self.num_epochs+1):
                start_time = time.time()
                num_examples = X_train.shape[0]
                losses = []

                shuffle_idxs = np.arange(X_train.shape[0])
                np.random.shuffle(shuffle_idxs)
                X_train = X_train[shuffle_idxs]
                S_train = S_train[shuffle_idxs]
                y_train = y_train[shuffle_idxs]

                for offset in xrange(0, num_examples-self.batch_size, self.batch_size):
                    end = min(offset+self.batch_size, num_examples)
                    for celltype_idx in xrange(y.shape[1]):
                        batch_sequence = np.reshape(X_train[offset:end, :],
                                                    (self.batch_size, self.height, self.width, self.num_channels))
                        batch_shapes = np.reshape(S_train[offset:end, :],
                                                  (self.batch_size, self.height, self.width, self.num_shape_features))
                        batch_labels = np.reshape(y_train[offset:end, celltype_idx], (self.batch_size, 1))
                        feed_dict = {self.tf_sequence: batch_sequence,
                                     self.tf_shape: batch_shapes,
                                     self.tf_train_labels: batch_labels,
                                     self.tf_gene_expression:
                                         np.reshape(gene_expression[celltype_idx],
                                                    (1, self.num_genexpr_features)),
                                     self.keep_prob: 1-self.dropout_rate
                                     }
                        if np.random.rand() <= 0.9 or celltype_idx != 0:
                            _, r_loss = session.run([optimizer, loss], feed_dict=feed_dict)
                        else:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()

                            summary, _, r_loss = session.run([self.summary_op, optimizer, loss],
                                                             feed_dict=feed_dict,
                                                             options=run_options,
                                                             run_metadata=run_metadata)
                            if self.model_dir is not None:
                                summary_writer.add_summary(summary)
                                summary_writer.add_run_metadata(run_metadata, 'Epoch %d, offset %d' % (epoch, offset))

                        losses.append(r_loss)
                losses = np.array(losses)
                t_loss = np.mean(losses)

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
                        batch_shapes = np.reshape(S_val[offset:end, :],
                                                  (self.batch_size, self.height, self.width, self.num_shape_features))
                        batch_labels = np.reshape(y_val[offset:end, celltype_idx], (self.batch_size, 1))
                        feed_dict = {self.tf_sequence: batch_sequence,
                                     self.tf_shape: batch_shapes,
                                     self.tf_train_labels: batch_labels,
                                     self.tf_gene_expression:
                                         np.reshape(gene_expression[celltype_idx],
                                                    (1, self.num_genexpr_features)),
                                     self.keep_prob: 1,
                                     }
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
                elif early_score == 1:
                    if self.verbose:
                        print "%d\t%f\t%f\t%.2f%%\t\t%ds" % \
                              (epoch, float(t_loss), float(v_loss), float(v_acc), int(time.time() - start_time))
                elif early_score == 0:
                    if self.verbose:
                        print "Early stopping triggered, exiting..."
                        break

            summary_writer.add_graph(session.graph)

if __name__ == '__main__':
    transcription_factor = 'CTCF'
    datareader = DataReader('../data/')
    datapath = '../data/'
    '''
            Run cross celltype benchmark for specified model and model_parameters
            :param datapath: path to data directory
            :param model: Model to be used
            :return:
            '''
    print "Running cross celltype benchmark for transcription factor %s" % transcription_factor
    # --------------- TRAIN
    celltypes = datareader.get_celltypes_for_tf(transcription_factor)

    self.num_train_instances = datareader.get_num_bound_lines(transcription_factor) * 2
    if len(celltypes) <= 1:
        print 'cant build cross transcription_factor validation for this transcription factor'
        return

    celltypes_train = celltypes[:-1]
    celltypes_test = celltypes[-1]

    if os.path.exists(
            os.path.join(datapath, 'models/' + transcription_factor + model.__class__.__name__ + 'Xtrain.npy')):
        X_train, S_train, y_train, y_train_val = load_model(transcription_factor, model)
    else:
        y_train_val = np.zeros((self.num_train_instances,), dtype=np.int32)
        X_train = self.get_X(model, self.num_train_instances)
        S_train = self.get_S(model, self.num_train_instances)
        y_train = self.get_y_train(model, self.num_train_instances, len(celltypes_train))

        for idx, instance in enumerate(self.datareader.generate_cross_celltype(transcription_factor,
                                                                               celltypes,
                                                                               [CrossvalOptions.balance_peaks])):
            if idx >= self.num_train_instances:
                break
            (_, _), sequence, sequence_features, shape_features, labels = instance
            X_train[idx] = self.get_X_data(model, sequence, sequence_features)
            S_train[idx] = self.get_S_data(model, shape_features)
            y_train[idx] = self.get_y_train_data(model, labels)
            y_train_val[idx] = labels.flatten()[-1]

    if save_train_set:
        self.save_model(transcription_factor, model, X_train, S_train, y_train, y_train_val)

    gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes_train)
    model.fit(X_train, y_train, S_train, gene_expression_features)