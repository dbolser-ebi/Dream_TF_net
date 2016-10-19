from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from performance_metrics import *
import tensorflow as tf
from keras.callbacks import Callback
tf.python.control_flow_ops = tf
from datagen import *

class ValidCallBack(Callback):
    def __init__(self, celltype, y_valid):
        super(ValidCallBack, self).__init__()
        self.celltype = celltype
        self.y_valid = y_valid

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.celltype, 'train', True)
        print_results(self.y_valid, predictions)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class KConvNet:
    def __init__(self, sequence_bin_size=200, num_epochs=1, batch_size=512, num_channels=4, verbose=False, config=7, dnase_bin_size=200):
        self.sequence_bin_size = sequence_bin_size
        self.num_epochs = num_epochs
        self.datagen = DataGenerator()
        self.transcription_factor = 'RFX5'
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.config = config
        self.verbose = verbose
        self.dnase_bin_size = dnase_bin_size
        self.bin_size = 200

        self.model = self.get_model()

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def get_model(self):
        sequence_model = Sequential()
        sequence_model.add(Convolution1D(15, 15, input_shape=(self.sequence_bin_size, self.num_channels)))
        sequence_model.add(MaxPooling1D(35, 35))
        sequence_model.add(Activation('relu'))
        sequence_model.add(Flatten())
        sequence_model.add(Dense(100, activation="relu"))

        if self.config == 1:
            sequence_model.add(Dense(1, activation='sigmoid'))
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
        model.add(Dense(1, activation='sigmoid'))
        model.compile(Adam(0.001), 'binary_crossentropy')

        return model

    def generate_batches(self, celltypes_train):
        trans_f_idx = self.datagen.get_trans_f_lookup()[self.transcription_factor]

        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2
        ids = np.array(range(self.datagen.train_length))
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
            self.datagen.train_length,
            self.num_epochs,
            1 if self.verbose else 0,
            class_weight={0: 1.0, 1: ratio},
            max_q_size=10,
            nb_worker=6,
            callbacks=[ValidCallBack(celltype_test, y_valid)]
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
