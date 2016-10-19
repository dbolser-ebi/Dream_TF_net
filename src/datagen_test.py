import unittest
from datagen import *
import numpy as np


class TestDataGen(unittest.TestCase):

    def test_indices_for_ids(self):
        datagen = DataGenerator()
        correction = 100
        X = np.load(os.path.join(datagen.save_dir, 'sequence_' + 'train' + '.npy'))
        ids = range(datagen.train_length)
        indices = datagen.get_positions_for_ids(ids, 'train')
        with gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:
            for i, line in enumerate(fin):
                if np.random.rand() > 0.999:
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])
                    sequence = datagen.hg19[chromosome][start-correction:end+correction]
                    sequence_one_hot = datagen.sequence_to_one_hot(np.array(list(sequence)))
                    index = indices[i]
                    assert(np.array_equal(sequence_one_hot, X[index-correction:index+200+correction]) == True)

    def test_y(self):
        datagen = DataGenerator()

        tf_idx = datagen.get_trans_f_lookup()['CTCF']

        y = datagen.get_y('HepG2')[:, tf_idx]

        with gzip.open('../data/chipseq_labels/CTCF.train.labels.tsv.gz') as fin:
            fin.readline()
            for idx, line in enumerate(fin.readlines()):
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                val = tokens[6]
                if val == 'B':
                    assert(y[idx] == 1)
                else:
                    assert(y[idx] == 0)





if __name__ == '__main__':
    unittest.main()
