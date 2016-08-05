import unittest
from datareader import *


class test_datareader(unittest.TestCase):
    def setUp(self):
        self.datareader = DataReader('../data/')

    def test_get_DNAse_peaks_tree(self):
        celltype = 'A549'
        peaks = [['chr17', '1617951', '1618762'],
['chr18', '56455911', '56457025'],
['chr1', '154946623', '154947606'],
['chr10', '74036321', '74037371'],
['chr19', '13961706', '13962482'],
['chr8', '62624667', '62625674'],
['chr7', '134049951', '134050773'],
['chr5', '172192031', '172194149'],
['chr6', '32937245', '32938256'],
['chr20', '52723831', '52725155']]
        peaks_tree = self.datareader.get_DNAse_peaks_tree(celltype)
        for (chromosome, start, stop) in peaks:
            self.assertTrue(DNasePeakEntry(chromosome, start) in peaks_tree)


if __name__ == '__main__':
    unittest.main()
