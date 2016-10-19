import tensorflow as tf
from keras.models import Sequential
from keras.layers import Merge, Convolution1D, MaxPooling1D, Activation, Flatten, Dense, Dropout
from datagen import *
from keras.optimizers import Adam


class KMultiConvNet:

    def __init__(self, config=7, bin_size=200, num_chunks=10, verbose=False, num_channels=4, num_epochs=1, batch_size=512):
        self.config = config
        self.bin_size = bin_size
        self.tf_ratio = tf.placeholder(dtype=tf.float32)
        self.datagen = DataGenerator()
        self.num_chunks = num_chunks
        self.num_epochs = num_epochs
        self.segment = 'train'
        self.num_channels = num_channels
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = self.get_model()

    def set_segment(self, segment):
        self.segment = segment

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

    def get_model(self):
        sequence_model = Sequential()
        sequence_model.add(Convolution1D(32, 15, input_shape=(self.bin_size, self.num_channels)))
        sequence_model.add(MaxPooling1D(35, 35))
        sequence_model.add(Activation('relu'))
        sequence_model.add(Flatten())
        sequence_model.add(Dense(100, activation="relu"))

        dnase_model = Sequential()
        dnase_model.add(Dense(100, input_dim=10, activation="relu"))

        model = Sequential()
        model.add(Merge([sequence_model, dnase_model], "concat"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='sigmoid'))
        model.compile(Adam(), self.get_loss)

        return model

    def generate_batch(self, celltype):
        while True:
            for chunk_id in range(1, self.num_chunks + 1):
                ids = range((chunk_id - 1) * 1000000, min(chunk_id * 1000000, self.datagen.train_length))
                X = self.datagen.get_sequece_from_ids(ids, self.segment, self.bin_size)
                y = self.datagen.get_y(celltype)[ids]
                dnase_features = self.datagen.get_dnase_features_from_ids(ids,
                                                                          self.segment,
                                                                          celltype,
                                                                          200)
                num_examples = len(ids)
                for i in range(0, num_examples, self.batch_size):
                    yield ([X[i:i+self.batch_size], dnase_features[i:i+self.batch_size]], y[i:i+self.batch_size])

    def fit(self, celltypes_train):
        self.model.fit_generator(self.generate_batch(celltypes_train),
                                 min(self.datagen.train_length, self.num_chunks*1000000),
                                 1, verbose=1 if self.verbose else 0)

    def predict(self, celltype, validation=False):
        '''
                Run trained model
                :return: predictions
                '''
        dnase_features_total = self.datagen.get_dnase_features(self.segment, celltype, self.bin_size)

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

        for start in range(0, num_test_indices, stride):
            ids = range(start, min(start + stride, num_test_indices))
            X = self.datagen.get_sequece_from_ids(ids, self.segment, self.bin_size)
            dnase_features = dnase_features_total[ids]

            prediction = self.model.predict((X, dnase_features), batch_size=1024)
            predictions.extend(prediction)
        predictions = np.array(predictions)
        return predictions
