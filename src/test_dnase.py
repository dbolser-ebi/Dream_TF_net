from datagen import *
import gzip
import pyBigWig
datagen = DataGenerator()

celltypes = ['GM12878', 'H1-hESC', 'HeLa-S3', 'K562', 'SK-N-SH']

for celltype in celltypes:
    dnase = datagen.get_dnase_features('train', celltype, 200)
    with gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:
        bw = pyBigWig.open("../data/debug/dnase_%s.bw" % celltype, "w")
        bw.addHeader([("chr10", 135534747)], maxZooms=0)
        for idx, line in enumerate(fin):
            if idx >= 2702470:
                break
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            bw.addEntries(["chr10"]*50, range(start, start+50), ends=range(start+1, start+51),
                          values=dnase[idx, :50].tolist())
        bw.close()