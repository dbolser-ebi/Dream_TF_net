import unittest
from datagen import *
from datareader import *
from sys import stdout


class TestBoleyScores(unittest.TestCase):
    def test_scores(self):
        datagen = DataGenerator()
        scores = datagen.get_boley_scores('train', 200, 'TAF1')
        pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
