import time
from datagen import *
np.random.seed(14522124)
import random
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
from keras.utils.visualize_util import plot
from keras.callbacks import *
from performance_metrics import *



class VProgbarLogger(Callback):
    def __init__(self, val_seq, val_dnase, val_labels, config, verbose):
        super(VProgbarLogger, self).__init__()
        self.val_seq = val_seq
        self.val_dnase = val_dnase
        self.val_labels = val_labels
        self.config = config
        self.verbose = verbose
        self.best_auprc = -1
        self.stagnant_steps = 0
        self.patience = 5

    '''Callback that prints metrics to stdout.
    '''
    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
            self.progbar = Progbar(target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        if batch % 100 == 0:
            if self.config == 2:
                predictions = self.model.predict([self.val_seq, self.val_dnase], batch_size=512, verbose=0)
            elif self.config == 1:
                predictions = self.model.predict(self.val_seq, batch_size=512, verbose=0)

            auc = auroc(self.val_labels, predictions)
            aup = average_precision_score(self.val_labels, predictions)

            self.log_values.append(('auroc', auc))
            self.log_values.append(('auprc', aup))


        # skip progbar update for the last batch;
        # will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['nb_sample']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)

        if self.config == 2:
            predictions = self.model.predict([self.val_seq, self.val_dnase], batch_size=512, verbose=0)
        elif self.config == 1:
            predictions = self.model.predict(self.val_seq, batch_size=512, verbose=0)

        auc = auroc(self.val_labels, predictions)
        aup = average_precision_score(self.val_labels, predictions)

        if aup > self.best_auprc:
            self.stagnant_steps = 0
            self.best_auprc = aup
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps > self.patience:
                self.model.stop_training = True


class KConvNet:
    def __init__(self, sequence_bin_size=200, num_epochs=1, batch_size=512,
                 num_channels=4, verbose=False, config=7, dnase_bin_size=200, randomize_celltypes=False,
                 regression = False):
        self.sequence_bin_size = sequence_bin_size
        self.num_epochs = num_epochs
        self.datagen = DataGenerator()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.config = config
        if self.config not in [1, 2]:
            self.config = 2
        self.verbose = verbose
        self.dnase_bin_size = dnase_bin_size
        self.bin_size = 200
        self.sequence_all = np.load(os.path.join(self.datagen.save_dir, 'sequence_' + 'train' + '.npy'))
        self.randomize_celltypes = randomize_celltypes
        self.regression = regression
        self.shared_model_reg = self.get_shared_model()
        self.shared_model_cls = self.get_shared_model()
        self.regression_model = self.get_model(shared_model=self.shared_model_reg, regression=True)
        self.model = self.get_model(self.shared_model_cls)

    def get_functional_model(self):
        # Sequence model
        sequence = Input(shape=(self.sequence_bin_size, self.num_channels), name='sequence')
        conv1 = Convolution1D(32, 15)(sequence)
        maxp1 = MaxPooling1D(35, 35)(conv1)
        batchnorm1 = BatchNormalization()(maxp1)
        activation1 = Activation('relu')(batchnorm1)
        flatten = Flatten()(activation1)
        drop1 = Dropout(0.5)(flatten)
        dense1 = Dense(100)(drop1)
        batchnorm2 = BatchNormalization()(dense1)
        activation2 = Activation('relu')(batchnorm2)
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

        output_dense_classification = Activation('sigmoid', name='regression')(Dense(1, name='classification')(drop))
        output_dense_regression = Dense(1)(drop)

        model = Model(input=[sequence, dnase], output=[output_dense_classification, output_dense_regression])

        model.compile(optimizer=Adam(),
                      loss={'classification': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'})
                      #loss_weights={'main_output': 1., 'aux_output': 0.2})

    def get_shared_model(self):

        sequence_model = Sequential()
        sequence_model.add(Convolution1D(32, 20, input_shape=(self.sequence_bin_size, self.num_channels)))
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
            sequence_model.add(Dense(1))
            return sequence_model

        dnase_model = Sequential()
        # dnase_model.add(Convolution1D(15, 50, input_shape=(self.dnase_bin_size, 1), subsample_length=50))
        dnase_model.add(MaxPooling1D(20, 20, input_shape=(self.dnase_bin_size, 1)))
        # dnase_model.add(BatchNormalization())
        # dnase_model.add(Activation('relu'))
        dnase_model.add(Flatten())
        # dnase_model.add(Dropout(0.5))
        # dnase_model.add(Dense(100, W_regularizer=l2(0.01)))
        # dnase_model.add(BatchNormalization())
        # dnase_model.add(Activation('relu'))
        # dnase_model.add(Dropout(0.5))

        if self.config == 2:
            model = Sequential()
            model.add(Merge([sequence_model, dnase_model], "concat"))
            model.add(Dense(1000))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            return model

    def get_model(self, shared_model, regression=False):
        shared_model.add(Dense(1))
        shared_model.add(BatchNormalization())
        if regression:
            shared_model.compile(Adam(), 'mse')
        else:
            shared_model.add(Activation('sigmoid'))
            shared_model.compile(Adam(), 'binary_crossentropy')
        return shared_model

    def get_validation_sample(self, celltype, transcription_factor):
        np.random.seed(14522124)
        trans_f_idx = self.datagen.get_trans_f_lookup()[transcription_factor]
        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2
        bound_ids = self.datagen.get_bound_positions_for_trans_f_celltype(transcription_factor, celltype)
        dnase_ids = self.datagen.get_dnase_accesible_ids_for_celltype(celltype, conservative=True)
        random_ids = range(self.datagen.train_length)
        random.shuffle(random_ids)
        ids = np.unique(np.array(bound_ids[:10000] + dnase_ids[:10000] + random_ids[:10000]))
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

    def generate_functional_batches(self, celltypes_train, transcription_factor, combination_ids):
        return

    def generate_precise_batches(self, celltypes_train, transcription_factor, combination_ids, regression=False):
        trans_f_idx = self.datagen.get_trans_f_lookup()[transcription_factor]
        celltype_lookup = {}

        sequence_bin_correction = (self.sequence_bin_size - self.bin_size) / 2
        dnase_bin_correction = (self.dnase_bin_size - self.bin_size) / 2

        start_positions = np.array(self.datagen.get_positions_for_ids(map(lambda (x, _): x, combination_ids), 'train'))

        labels_all = []
        dnase_all = []
        for c_idx, celltype in enumerate(celltypes_train):
            dnase_track = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % ('train', celltype)))
            dnase_all.append(dnase_track)
            if regression:
                labels = self.datagen.get_chipseq_signal(celltype)
            else:
                labels = self.datagen.get_labels(celltype)
            labels_all.append(labels)
            celltype_lookup[celltype] = c_idx

        while True:
            for use_complement in [False]:
                for i in range(0, len(combination_ids), self.batch_size):
                    start_positions_batch = start_positions[i:i + self.batch_size]
                    batch_sequence = np.zeros((start_positions_batch.size, self.sequence_bin_size, self.num_channels),
                                              dtype=np.float32)
                    batch_dnase = np.zeros((start_positions_batch.size, self.dnase_bin_size, 1), dtype=np.float32)
                    batch_labels = np.zeros((start_positions_batch.size,), dtype=np.float32)
                    for j, index in enumerate(start_positions_batch):
                        celltype_idx = celltype_lookup[combination_ids[i+j][1]]
                        sequence_sl = slice(index - sequence_bin_correction,
                                            index + self.bin_size + sequence_bin_correction)
                        batch_sequence[j] = self.sequence_all[sequence_sl]
                        if use_complement:
                            batch_sequence[j] = self.datagen.get_complement(batch_sequence[j])

                        dnase_sl = slice(index - dnase_bin_correction, index + self.bin_size + dnase_bin_correction)
                        batch_dnase[j] = np.reshape(dnase_all[celltype_idx][dnase_sl], (-1, 1))
                        batch_labels[j] = labels_all[celltype_idx][combination_ids[i+j][0], trans_f_idx]

                    batch_dnase = np.log(batch_dnase + 1)

                    if self.config == 1:
                        yield (batch_sequence, batch_labels)
                    elif self.config == 2:
                        yield ([batch_sequence, batch_dnase], batch_labels)

    def generate_batches(self, celltypes_train, transcription_factor, ids):
        trans_f_idx = self.datagen.get_trans_f_lookup()[transcription_factor]
        celltype_lookup = self.datagen.get_celltype_lookup()

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
            '''
            shuffle_idx = np.arange(len(ids))
            np.random.shuffle(shuffle_idx)
            shuffled_ids = ids[shuffle_idx]
            shuffled_start_positions = start_positions[shuffle_idx]
            '''
            for c_idx, celltype in enumerate(celltypes_train):
                if not self.randomize_celltypes:
                    dnase = np.load(os.path.join(self.datagen.save_dir, 'dnase_fold_%s_%s.npy' % ('train', celltype)))
                    labels = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
                for i in range(0, len(ids), self.batch_size):
                    celltype_idx = np.random.randint(0, len(celltypes_train))
                    start_positions_batch = start_positions[i:i + self.batch_size]
                    if self.randomize_celltypes:
                        batch_labels = labels_all[ids[i:i + self.batch_size], trans_f_idx, celltype_lookup[celltypes_train[celltype_idx]]].astype(np.float32)
                    else:
                        batch_labels = labels[ids[i: i+self.batch_size], trans_f_idx]
                    batch_sequence = np.zeros((start_positions_batch.size, self.sequence_bin_size, self.num_channels), dtype=np.float32)
                    batch_dnase = np.zeros((start_positions_batch.size, self.dnase_bin_size, 1), dtype=np.float32)
                    for j, index in enumerate(start_positions_batch):
                        sequence_sl = slice(index-sequence_bin_correction, index+self.bin_size+sequence_bin_correction)
                        batch_sequence[j] = self.sequence_all[sequence_sl]
                        dnase_sl = slice(index-dnase_bin_correction, index+self.bin_size+dnase_bin_correction)
                        if self.randomize_celltypes:
                            batch_dnase[j] = np.reshape(dnase_all[celltype_idx][dnase_sl], (-1, 1))
                        else:
                            batch_dnase[j] = np.reshape(dnase[dnase_sl], (-1, 1))

                    batch_dnase = np.log(batch_dnase+1)

                    if self.config == 1:
                        yield (batch_sequence, batch_labels)
                    elif self.config == 2:
                        yield ([batch_sequence, batch_dnase], batch_labels)

    def fit(self, celltypes_train, transcription_factor, celltype_test=None):
        np.random.seed(14522124)
        trans_f_idx = self.datagen.get_trans_f_lookup()[transcription_factor]
        #val_seq, val_dnase, val_lab = self.get_validation_sample(celltype_test, transcription_factor)
        full_ids = range(self.datagen.train_length)
        random.shuffle(full_ids)

        combination_ids = []
        dnase_ids_per_celltype = []
        num_samples = 0
        for celltype in celltypes_train:
            dnase_ids = full_ids[:100000] + self.datagen.get_dnase_accesible_ids([celltype], conservative=True)
            random.shuffle(dnase_ids)
            dnase_ids_per_celltype.append(dnase_ids)
            for id in dnase_ids:
                combination_ids.append((id, celltype))
                num_samples += 1

        num_bound = 0
        num_unbound = 0
        for c_idx, celltype in enumerate(celltypes_train):
            y = self.datagen.get_y(celltype)[dnase_ids_per_celltype[c_idx], trans_f_idx]
            num_bound += y[y == 1].size
            num_unbound += y[y == 0].size

        ratio = num_unbound / num_bound

        if self.regression:
            if self.randomize_celltypes:
                random.shuffle(combination_ids)
            history = self.regression_model.fit_generator(
                self.generate_precise_batches(celltypes_train, transcription_factor, combination_ids, regression=True),
                samples_per_epoch=len(combination_ids),  # len(ids)*len(celltypes_train),
                nb_epoch=self.num_epochs,
                verbose=0,
                max_q_size=10,
                nb_worker=1,
                #callbacks=[VProgbarLogger(val_seq, val_dnase, val_lab, self.config, 1 if self.verbose else 0)],
                pickle_safe=False
            )
            print history.history
            self.shared_model_cls.set_weights(self.shared_model_reg.get_weights())

        if self.randomize_celltypes:
            random.shuffle(combination_ids)
        history = self.model.fit_generator(
            self.generate_precise_batches(celltypes_train, transcription_factor, combination_ids),
            samples_per_epoch=len(combination_ids),#len(ids)*len(celltypes_train),
            nb_epoch=self.num_epochs,
            verbose=0,
            class_weight={0: 1.0, 1: ratio},
            max_q_size=10,
            nb_worker=1,
            #callbacks=[VProgbarLogger(val_seq, val_dnase, val_lab, self.config, 1 if self.verbose else 0)],
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

        test_batch_size = self.batch_size

        for i in range(0, len(ids), test_batch_size):
            #start_time = time.time()
            start_positions_batch = start_positions[i:i + test_batch_size]
            batch_sequence = np.zeros((len(start_positions_batch), self.sequence_bin_size, self.num_channels), dtype=np.float32)
            batch_dnase = np.zeros((len(start_positions_batch), self.dnase_bin_size, 1), dtype=np.float32)
            for j, index in enumerate(start_positions_batch):
                sequence_sl = slice(index - sequence_bin_correction,
                                    index + self.bin_size + sequence_bin_correction)
                batch_sequence[j] = self.sequence_all[sequence_sl]
                dnase_sl = slice(index - dnase_bin_correction, index + self.bin_size + dnase_bin_correction)
                batch_dnase[j] = np.reshape(dnase[dnase_sl], (-1, 1))

            batch_dnase = np.log(batch_dnase + 1)

            if self.config == 1:
                yield batch_sequence
            elif self.config == 2:
                yield [batch_sequence, batch_dnase]
        time.sleep(100000)

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
