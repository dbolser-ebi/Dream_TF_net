import unittest
from datagen import *
from datareader import *
from sys import stdout


class TestSequenceExtraction(unittest.TestCase):

    '''
    def test_sequences(self):
        reader = DataReader('../data/')
        hg19 = Fasta('../data/annotations/hg19.genome.fa')
        X = np.load('../data/preprocess/features/segment_train_bin_size_600_chunk_id_0.npy')
        chunk_limit = 1000000

        with gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:
            bin_size = 600
            bin_correction = max(0, (bin_size - 200) / 2)
            for idx, line in enumerate(fin):
                if idx >= chunk_limit:
                    X = np.load('../data/preprocess/features/segment_train_bin_size_600_chunk_id_%d.npy' % chunk_limit)
                    chunk_limit += 1000000

                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                sequence = hg19[chromosome][start - bin_correction:start + 200 + bin_correction]
                valid = reader.sequence_to_one_hot(np.array(list(sequence)))
                current = X[idx % 1000000]
                self.assertTrue((valid == current).all())
    '''

    def test_get_sequece_from_ids(self):
        hg19 = Fasta('../data/annotations/hg19.genome.fa')
        reader = DataReader('../data/')
        datagen = DataGenerator()
        ids = random.sample(xrange(51676736), 1000)
        X = datagen.get_sequece_from_ids(ids, 'train')

        with gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:
            bin_size = 600
            bin_correction = max(0, (bin_size - 200) / 2)

            for idx, line in enumerate(fin):
                if idx in ids:
                    print idx
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])
                    sequence = hg19[chromosome][start - bin_correction:start + 200 + bin_correction]
                    valid = reader.sequence_to_one_hot(np.array(list(sequence)))
                    current = X[ids.index(idx)]
                    self.assertTrue((valid == current).all())

                if idx > max(ids):
                    break

if __name__ == '__main__':
    unittest.main()
