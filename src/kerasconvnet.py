from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Activation, Flatten, Dense, Merge, Dropout
from keras.optimizers import Adam
from datagen import *


class KConvNet:
    def __init__(self, bin_size, num_epochs=1, num_chunks=10, batch_size=512, num_channels=4, verbose=False, config=7):
        self.bin_size = bin_size
        self.num_epochs = num_epochs
        self.num_chunks = num_chunks
        self.datagen = DataGenerator()
        self.transcription_factor = 'RFX5'
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.config = config
        self.verbose = 1 if verbose else 0

        self.model = self.get_model()

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def get_model(self):

        sequence_model = Sequential()
        sequence_model.add(Convolution1D(32, 30, init='uniform', activation='linear', input_shape=(self.bin_size, self.num_channels)))
        sequence_model.add(MaxPooling1D(35, 35))
        sequence_model.add(Activation('relu'))
        sequence_model.add(Flatten())
        sequence_model.add(Dense(100, activation="relu"))

        if self.config == 1:
            sequence_model.add(Dense(1, activation='sigmoid'))
            sequence_model.compile(Adam(), 'binary_crossentropy', metrics=['accuracy'])
            return sequence_model

        dnase_model = Sequential()
        dnase_model.add(Dense(100, input_dim=10, activation="relu"))

        model = Sequential()
        model.add(Merge([sequence_model, dnase_model], "concat"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(Adam(), 'binary_crossentropy', metrics=['binary_accuracy', 'matthews_correlation'])

        return model

    def fit_combined(self, celltypes_train, celltype_test):
        trans_f_idx = self.datagen.get_trans_f_lookup()[self.transcription_factor]

        dnase_features_test = self.datagen.get_dnase_features('train', celltype_test, 200)
        y_test_for_tf = self.datagen.get_y(celltype_test)[:, trans_f_idx]

        for c_idx, celltype in enumerate(celltypes_train):
            y = self.datagen.get_y(celltype)[:, trans_f_idx]
            dnase_features = self.datagen.get_dnase_features('train', celltype, 200)

            print 'Data for celltype', celltype, 'loaded.'

            for chunk_id in range(1, self.num_chunks + 1):
                print "Running chunk %d" % chunk_id
                ids = range((chunk_id - 1) * 1000000, min(chunk_id * 1000000, self.datagen.train_length))

                X = self.datagen.get_sequece_from_ids(ids, 'train', self.bin_size)
                y_chunk = y[ids]
                y_chunk_test = y_test_for_tf[ids]
                dnase_chunk = dnase_features[ids]
                dnase_chunk_test = dnase_features_test[ids]

                # Batch stratification and shuffling

                bound_idxs = np.where(y_chunk == 1)[0]
                unbound_idxs = np.where(y_chunk == 0)[0]

                chunk_ratio = unbound_idxs.shape[0] / bound_idxs.shape[0]
                if self.config == 1:
                    history = self.model.fit(X, y_chunk,
                                             batch_size=self.batch_size, nb_epoch=self.num_epochs,
                                             class_weight={0: 1, 1: chunk_ratio},
                                             validation_data=(X, y_chunk_test), verbose=self.verbose)
                else:
                    history = self.model.fit([X, dnase_chunk], y_chunk,
                                             batch_size=self.batch_size, nb_epoch=self.num_epochs,
                                             class_weight={0: 1, 1: chunk_ratio},
                                             validation_data=([X, dnase_chunk_test], y_chunk_test), verbose=self.verbose)
                print history.history

    def predict_combined(self, X, dnase_features):
        '''
        Run trained model
        :return: predictions
        '''
        predictions = self.model.predict([X, dnase_features], batch_size=1024)
        return predictions
