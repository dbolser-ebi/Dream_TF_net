import tensorflow as tf
from keras.models import *
from keras.layers import *
from datagen import *
from keras.optimizers import Adam
import time
tf.python.control_flow_ops = tf


class KMultiConvNet:

    def __init__(self, config=2, bin_size=200, verbose=False, num_channels=4, num_epochs=1, batch_size=512,
                 sequence_bin_size=200, dnase_bin_size=200, randomize_celltypes=False):
        self.config = config
        self.bin_size = bin_size
        self.tf_ratio = tf.placeholder(dtype=tf.float32)
        self.datagen = DataGenerator()
        self.num_epochs = num_epochs
        self.segment = 'train'
        self.num_channels = num_channels
        self.verbose = verbose
        self.batch_size = batch_size
        self.sequence_bin_size = sequence_bin_size
        self.dnase_bin_size = dnase_bin_size
        self.randomize_celltypes = randomize_celltypes
        self.sequence_all = np.load(os.path.join(self.datagen.save_dir, 'sequence_' + 'train' + '.npy'))
        self.num_transcription_factors = 32
        self.model = self.get_multi_model() #self.get_model()

    '''

    def get_loss(self, labels, logits):
        logits = tf.reshape(logits, [-1])
        labels = tf.reshape(labels, [-1])
        index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))
        logits_known = tf.gather(logits, index)
        labels_known = tf.gather(labels, index)
        entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits_known, labels_known)
        return tf.reduce_mean(entropies)

    def calc_loss_seperate(self, labels, logits):
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
        loss_bound = tf.mul(self.tf_ratio, tf.cond(tf.equal(num_bound, tf.constant(0.0)), lambda: tf.constant(0.0),
                                           lambda: tf.div(entropy_bound, num_unbound)))
        loss_unbound = tf.div(entropy_unbound, num_unbound)
        return tf.add(loss_bound, loss_unbound)

    '''

    def get_validation_sample(self, celltype):
        np.random.seed(14522124)
        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2
        random_ids = range(self.datagen.train_length)
        random.shuffle(random_ids)
        ids = np.array(random_ids[:10000])
        start_positions = np.array(self.datagen.get_positions_for_ids(ids, 'train'))

        sequence_bins = np.zeros((ids.size, self.sequence_bin_size, self.num_channels), dtype=np.float16)
        dnase_bins = np.zeros((ids.size, self.dnase_bin_size, 1), dtype=np.float16)
        label_bins = np.zeros((ids.size,), dtype=np.float16)

        dnase = self.datagen.get_dnase('train', celltype)
        labels = self.datagen.get_labels(celltype)[ids, :]

        for i, index in enumerate(start_positions):
            sequence_bins[i] = self.sequence_all[
                               index - sequence_bin_correction: index + self.bin_size + sequence_bin_correction]
            dnase_bins[i] = np.reshape(
                dnase[index - dnase_bin_correction: index + self.bin_size + dnase_bin_correction], (-1, 1))
            label_bins[i] = labels[i]

        return sequence_bins, dnase_bins, label_bins

    def get_multi_model(self):
        reverse_trans_f_lookup = self.datagen.get_reverse_trans_f_lookup()

        # Sequence model

        sequence = Input(shape=(self.sequence_bin_size, self.num_channels), name='sequence')
        conv1 = Convolution1D(
                        nb_filter=320,
                        filter_length=30)(sequence)
        maxp1 = MaxPooling1D(pool_length=15, stride=15)(conv1)
        act1 = Activation('relu')(maxp1)
        drop1 = Dropout(0.2)(act1)
        #brnn = Bidirectional(LSTM(512))(drop1)
        flatten = Flatten()(drop1)
        #drop1 = Dropout(0.5)(brnn)
        dense1 = Dense(1000)(flatten)
        activation2 = Activation('relu')(dense1)
        sequence_model = Dropout(0.5)(activation2)

        # DNase model
        dnase = Input(shape=(self.dnase_bin_size, 1), name='dnase')
        maxp1 = MaxPooling1D(30, 30)(dnase)
        dnase_model = Flatten()(maxp1)

        # Merge
        merged = merge([sequence_model, dnase_model], mode='concat')
        dense = Dense(1000)(merged)
        batchnorm = BatchNormalization()(dense)
        activation = Activation('relu')(batchnorm)
        drop = Dropout(0.5)(activation)
        outputs = []
        for i in range(self.num_transcription_factors):
            dense = Dense(3)(drop)
            output = Activation('softmax', name='%s' % reverse_trans_f_lookup[i])(dense)
            outputs.append(output)

        # Objective
        model = Model(input=[sequence, dnase], output=outputs)
        model.compile(Adam(), 'categorical_crossentropy')

        return model

    def get_model(self):
        sequence_model = Sequential()
        sequence_model.add(Convolution1D(128, 15, input_shape=(self.sequence_bin_size, self.num_channels)))
        sequence_model.add(MaxPooling1D(35, 35))
        sequence_model.add(BatchNormalization())
        sequence_model.add(Activation('relu'))
        sequence_model.add(Flatten())
        sequence_model.add(Dropout(0.5))
        sequence_model.add(Dense(100))
        sequence_model.add(BatchNormalization())
        sequence_model.add(Activation('relu'))
        sequence_model.add(Dropout(0.5))

        if self.config == 1:
            sequence_model.add(Dropout(0.5))
            sequence_model.add(Dense(32, activation='sigmoid'))
            sequence_model.compile(Adam(), 'binary_crossentropy')
            return sequence_model

        dnase_model = Sequential()
        #dnase_model.add(Convolution1D(15, 50, input_shape=(self.dnase_bin_size, 1), subsample_length=50))
        dnase_model.add(MaxPooling1D(30, 30, input_shape=(self.dnase_bin_size, 1)))
        #dnase_model.add(BatchNormalization())
        #dnase_model.add(Activation('relu'))
        dnase_model.add(Flatten())
        #dnase_model.add(Dropout(0.5))
        #dnase_model.add(Dense(100, W_regularizer=l2(0.01)))
        #dnase_model.add(BatchNormalization())
        #dnase_model.add(Activation('relu'))
        #dnase_model.add(Dropout(0.5))

        if self.config == 2:
            model = Sequential()
            model.add(Merge([sequence_model, dnase_model], "concat"))
            model.add(Dense(1000))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32))
            model.add(BatchNormalization())
            model.add(Activation('sigmoid'))
            model.compile(Adam(0.001), 'binary_crossentropy')
            return model

    def get_validation_sample(self, celltype, transcription_factor):
        np.random.seed(14522124)
        trans_f_idx = self.datagen.get_trans_f_lookup()[transcription_factor]
        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2
        bound_ids = self.datagen.get_bound_positions_for_trans_f_celltype(transcription_factor, celltype)
        dnase_ids = self.datagen.get_dnase_accesible_ids_for_celltype(celltype, conservative=True)
        random_ids = range(self.datagen.train_length)
        random.shuffle(random_ids)
        ids = np.unique(np.array(bound_ids[:1000] + dnase_ids[:1000] + random_ids[:1000]))
        start_positions = np.array(self.datagen.get_positions_for_ids(ids, 'train'))

        sequence_bins = np.zeros((ids.size, self.sequence_bin_size, self.num_channels), dtype=np.float16)
        dnase_bins = np.zeros((ids.size, self.dnase_bin_size, 1), dtype=np.float16)
        label_bins = np.zeros((ids.size,), dtype=np.float16)

        dnase = self.datagen.get_dnase('train', celltype)
        labels = self.datagen.get_labels(celltype)[ids, trans_f_idx]

        for i, index in enumerate(start_positions):
            sequence_bins[i] = self.sequence_all[index - sequence_bin_correction: index+self.bin_size + sequence_bin_correction]
            dnase_bins[i] = np.reshape(dnase[index - dnase_bin_correction: index + self.bin_size + dnase_bin_correction], (-1,1))
            label_bins[i] = labels[i]

        return sequence_bins, dnase_bins, label_bins

    def generate_batches(self, celltypes_train, ids):
        celltype_lookup = self.datagen.get_celltype_lookup()
        reverse_trans_f_lookup = self.datagen.get_reverse_trans_f_lookup()

        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2

        start_positions = np.array(self.datagen.get_positions_for_ids(ids, 'train'))

        if self.randomize_celltypes:
            labels_all = np.load(os.path.join(self.datagen.save_dir, 'y_full.npy'))
            dnase_all = []
            for c_idx, celltype in enumerate(celltypes_train):
                dnase_track = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % ('train', celltype)))
                dnase_all.append(dnase_track)

        while True:
            shuffle_idx = np.arange(len(ids))
            np.random.shuffle(shuffle_idx)
            shuffled_ids = ids[shuffle_idx]
            shuffled_start_positions = start_positions[shuffle_idx]
            for c_idx, celltype in enumerate(celltypes_train):
                if not self.randomize_celltypes:
                    dnase = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % ('train', celltype)))
                    labels = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
                for i in range(0, len(ids), self.batch_size):
                    start_positions_batch = shuffled_start_positions[i:i + self.batch_size]
                    celltype_indexes = np.random.randint(0, len(celltypes_train), (start_positions_batch.size, ))

                    # BATCH PLACEHOLDERS
                    batch_labels_categories = np.zeros((start_positions_batch.size, self.num_transcription_factors, 3), dtype=np.float32)
                    batch_sequence = np.zeros((start_positions_batch.size, self.sequence_bin_size, self.num_channels),
                                              dtype=np.float32)
                    batch_dnase = np.zeros((start_positions_batch.size, self.dnase_bin_size, 1), dtype=np.float32)

                    if self.randomize_celltypes:
                        celltype_label_indexes = [celltype_lookup[celltypes_train[x]] for x in celltype_indexes]
                        batch_labels = labels_all[shuffled_ids[i:i + self.batch_size], :, celltype_label_indexes].astype(np.float32)
                    else:
                        batch_labels = labels[shuffled_ids[i: i+self.batch_size], :]

                    ukn_idx = np.where(batch_labels == -1)
                    unbound_idx = np.where(batch_labels == 0)
                    bound_idx = np.where(batch_labels == 1)
                    batch_labels_categories[ukn_idx[0], ukn_idx[1], 0] = 1
                    batch_labels_categories[unbound_idx[0], unbound_idx[1], 1] = 1
                    batch_labels_categories[bound_idx[0], bound_idx[1], 2] = 1

                    for j, index in enumerate(start_positions_batch):
                        sequence_sl = slice(index-sequence_bin_correction, index+self.bin_size+sequence_bin_correction)
                        batch_sequence[j] = self.sequence_all[sequence_sl]
                        dnase_sl = slice(index-dnase_bin_correction, index+self.bin_size+dnase_bin_correction)
                        if self.randomize_celltypes:
                            batch_dnase[j] = np.reshape(dnase_all[celltype_indexes[j]][dnase_sl], (-1, 1))
                        else:
                            batch_dnase[j] = np.reshape(dnase[dnase_sl], (-1, 1))

                    batch_dnase = np.log(batch_dnase+1)

                    feed_dict_input = {'sequence': batch_sequence, 'dnase': batch_dnase}

                    feed_dict_output = {}

                    for out_idx in range(self.num_transcription_factors):
                        feed_dict_output['%s' % reverse_trans_f_lookup[out_idx]] = batch_labels_categories[:, out_idx, :]

                    yield (feed_dict_input, feed_dict_output)

    def fit(self, celltypes_train, celltype_test=None):

        #dnase_ids = self.datagen.get_dnase_accesible_ids(celltypes_train, conservative=True)
        random_ids = range(self.datagen.train_length)
        random.shuffle(random_ids)
        ids = np.array(random_ids)#np.array(dnase_ids)

        num_bound = 0
        num_unbound = 0
        for celltype in celltypes_train:
            y = self.datagen.get_y(celltype)[ids, :]
            num_bound += y[y == 1].size
            num_unbound += y[y == 0].size
        ratio = num_unbound / num_bound

        history = self.model.fit_generator(
            self.generate_batches(celltypes_train, ids),
            len(ids),#*len(celltypes_train),
            self.num_epochs,
            class_weight={0: 0, 1: 1.0, 2: ratio},
            verbose=1 if self.verbose else 0,
            max_q_size=10,
            nb_worker=1,
            pickle_safe=False
        )
        print history.history

    def generate_test_batches(self, celltype, segment, validation=False):
        if segment == 'train':
            num_test_indices = 51676736
        if segment == 'ladder':
            num_test_indices = 8843011
        if segment == 'test':
            num_test_indices = 60519747
        if validation:
            num_test_indices = 2702470

        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2

        if segment == 'ladder' or segment == 'test':
            self.sequence_all = np.load(os.path.join(self.datagen.save_dir, 'sequence_' + segment + '.npy'))
        dnase = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % (segment, celltype)))

        ids = range(num_test_indices)
        start_positions = self.datagen.get_positions_for_ids(ids, segment)

        for i in range(0, len(ids), self.batch_size):
            start_positions_batch = start_positions[i:i + self.batch_size]
            batch_sequence = np.zeros((len(start_positions_batch), self.sequence_bin_size, self.num_channels), dtype=np.float32)
            batch_dnase = np.zeros((len(start_positions_batch), self.dnase_bin_size, 1), dtype=np.float32)
            for j, index in enumerate(start_positions_batch):
                sequence_sl = slice(index - sequence_bin_correction,
                                    index + self.bin_size + sequence_bin_correction)
                batch_sequence[j] = self.sequence_all[sequence_sl]
                dnase_sl = slice(index - dnase_bin_correction, index + self.bin_size + dnase_bin_correction)
                batch_dnase[j] = np.reshape(dnase[dnase_sl], (-1, 1))

            batch_dnase = np.log(batch_dnase + 1)

            feed_dict_input = {}
            feed_dict_input['sequence'] = batch_sequence
            feed_dict_input['dnase'] = batch_dnase

            yield feed_dict_input
        time.sleep(1000)

    def predict(self, celltype, segment, validation=False):
        '''
        Run trained model
        :return: predictions
        '''

        if segment == 'train':
            num_test_indices = 51676736
        if segment == 'ladder':
            num_test_indices = 8843011
        if segment == 'test':
            num_test_indices = 60519747
        if validation:
            num_test_indices = 2702470

        predictions = self.model.predict_generator(self.generate_test_batches(celltype, segment, validation),
                                                   num_test_indices, 10, 1, False)
        return predictions

