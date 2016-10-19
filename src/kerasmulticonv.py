import tensorflow as tf
from keras.models import Sequential
from keras.layers import Merge, Convolution1D, MaxPooling1D, Activation, Flatten, Dense, Dropout
from datagen import *
from keras.optimizers import Adam


class KMultiConvNet:

    def __init__(self, config=7, bin_size=200, verbose=False, num_channels=4, num_epochs=1, batch_size=512,
                 sequence_bin_size=200, dnase_bin_size=200):
        self.config = config
        self.bin_size = bin_size
        self.tf_ratio = tf.placeholder(dtype=tf.float32)
        self.datagen = DataGenerator()
        self.num_epochs = num_epochs
        self.segment = 'train'
        self.num_channels = num_channels
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = self.get_model()
        self.sequence_bin_size = sequence_bin_size
        self.dnase_bin_size = dnase_bin_size

    def set_segment(self, segment):
        self.segment = segment

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

    def get_model(self):
        sequence_model = Sequential()
        sequence_model.add(Convolution1D(15, 15, input_shape=(self.sequence_bin_size, self.num_channels)))
        sequence_model.add(MaxPooling1D(35, 35))
        sequence_model.add(Activation('relu'))
        sequence_model.add(Flatten())
        sequence_model.add(Dense(100, activation="relu"))

        if self.config == 1:
            sequence_model.add(Dense(32, activation='sigmoid'))
            sequence_model.compile(Adam(), 'binary_crossentropy')
            return sequence_model

        dnase_model = Sequential()
        dnase_model.add(Convolution1D(15, 15, input_shape=(self.dnase_bin_size, 1)))
        dnase_model.add(MaxPooling1D(35, 35))
        dnase_model.add(Activation('relu'))
        dnase_model.add(Flatten())
        dnase_model.add(Dense(100, activation="relu"))

        model = Sequential()
        model.add(Merge([sequence_model, dnase_model], "concat"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='sigmoid'))
        model.compile(Adam(0.001), 'binary_crossentropy')

        return model

    def generate_batches(self, celltypes_train):
        trans_f_idx = self.datagen.get_trans_f_lookup()[self.transcription_factor]

        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2
        ids = self.datagen.get_dnase_accesible_ids(celltypes_train, False)#np.array(range(self.datagen.train_length))
        start_positions = np.array(self.datagen.get_positions_for_ids(ids, 'train'))
        assert(ids.size == start_positions.size)
        sequence_all = np.load(os.path.join(self.datagen.save_dir, 'sequence_' + 'train' + '.npy'))

        labels_all = []
        dnase_all = []

        for c_idx, celltype in enumerate(celltypes_train):
            labels_all.append(self.datagen.get_y(celltype)[:, trans_f_idx])
            dnase_all.append(np.load(os.path.join(self.datagen.save_dir,
                                                        'dnase_fold_%s_%s.npy' % ('train', celltype))))
        while True:
            shuffle_idx = np.arange(len(ids))
            np.random.shuffle(shuffle_idx)
            shuffled_ids = ids[shuffle_idx]
            shuffled_start_positions = start_positions[shuffle_idx]

            for i in range(0, len(ids), self.batch_size):
                celltype_idx = np.random.randint(0, len(celltypes_train))

                start_positions_batch = shuffled_start_positions[i:i + self.batch_size]
                batch_labels = labels_all[celltype_idx][shuffled_ids[i:i + self.batch_size]]
                batch_sequence = np.zeros((start_positions_batch.size, self.sequence_bin_size, self.num_channels), dtype=np.float32)
                batch_dnase = np.zeros((start_positions_batch.size, self.dnase_bin_size, 1), dtype=np.float32)
                for j, index in enumerate(start_positions_batch):
                    sequence_sl = slice(index-sequence_bin_correction, index+self.bin_size+sequence_bin_correction)
                    batch_sequence[j] = sequence_all[sequence_sl]
                    dnase_sl = slice(index-dnase_bin_correction, index+self.bin_size+dnase_bin_correction)
                    batch_dnase[j] = np.reshape(dnase_all[celltype_idx][dnase_sl], (-1, 1))

                if self.config == 1:
                    yield (batch_sequence, batch_labels)
                else:
                    yield ([batch_sequence, batch_labels], batch_labels)

    def fit(self, celltypes_train, celltype_test):
        trans_f_idx = self.datagen.get_trans_f_lookup()[self.transcription_factor]

        valid_ids = range(2702470)
        y_valid = self.datagen.get_y(celltype_test)[valid_ids, trans_f_idx]

        num_bound = 0
        num_unbound = 0
        for celltype in celltypes_train:
            y = self.datagen.get_y(celltype)[:, trans_f_idx]
            num_bound += np.count_nonzero(y)
            num_unbound += y.shape[0] - num_bound

        ratio = num_unbound / num_bound

        self.model.fit_generator(
            self.generate_batches(celltypes_train),
            1000000,#self.datagen.train_length,
            self.num_epochs,
            1 if self.verbose else 0,
            class_weight={0: 1.0, 1: ratio},
            max_q_size=10,
            nb_worker=6,
        )

    def generate_test_batches(self, celltype, segment, validation=False):
        if segment == 'train':
            num_test_indices = 51676736
        if segment == 'ladder':
            num_test_indices = 8843011
        if segment == 'test':
            num_test_indices = 60519747
        if validation:
            num_test_indices = 2702470

        while True:
            sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
            dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2

            sequence_all = np.load(os.path.join(self.datagen.save_dir, 'sequence_' + segment + '.npy'))
            dnase = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % (segment, celltype)))

            ids = range(num_test_indices)
            start_positions = self.datagen.get_positions_for_ids(ids, segment)

            for i in range(0, len(ids), self.batch_size):
                start_positions_batch = start_positions[i:i + self.batch_size]
                batch_sequence = np.zeros((len(start_positions_batch), self.sequence_bin_size, self.num_channels))
                batch_dnase = np.zeros((len(start_positions_batch), self.dnase_bin_size, 1))
                for j, index in enumerate(start_positions_batch):
                    sequence_sl = slice(index - sequence_bin_correction,
                                        index + self.bin_size + sequence_bin_correction)
                    batch_sequence[j] = sequence_all[sequence_sl]
                    dnase_sl = slice(index - dnase_bin_correction, index + self.bin_size + dnase_bin_correction)
                    batch_dnase[j] = np.reshape(dnase[dnase_sl], (-1, 1))

                if self.config == 1:
                    yield batch_sequence
                else:
                    yield [batch_sequence, batch_dnase]

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
                                                   num_test_indices, 10, 6)
        return predictions

