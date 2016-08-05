import os
from datareader import *

dnase_cons_dir = '/run/media/niels/6d469c5d-2dc7-4079-b0bc-6880bb5c8387/ENCDREAMTFCHALLENGE/data/dnase_peaks_conservative'
dnase_rel_dir = '/run/media/niels/6d469c5d-2dc7-4079-b0bc-6880bb5c8387/ENCDREAMTFCHALLENGE/data/dnase_peaks_relaxed'
chipseq_label_dir = '/run/media/niels/6d469c5d-2dc7-4079-b0bc-6880bb5c8387/ENCDREAMTFCHALLENGE/data/chipseq_labels'

datareader = DataReader('../data/')

for tf in datareader.get_tfs():
    for celltype in datareader.get_celltypes_for_tf(tf):
        print 'tf', tf, 'celltype', celltype,
        f_label = gzip.open(os.path.join(chipseq_label_dir, tf+'.train.labels.tsv.gz'))
        f_dnase_c = gzip.open(os.path.join(dnase_cons_dir, 'DNASE.{}.conservative.narrowPeak.gz').format(celltype))
        f_dnase_r = gzip.open(os.path.join(dnase_rel_dir, 'DNASE.{}.relaxed.narrowPeak.gz').format(celltype))

        header = f_label.readline().split()
        for idx, celltype_ in enumerate(header[3:]):
            if celltype == celltype_:
                break

        ordering = {'chr10':0,
                    'chr1':1,
                    'chr11':2,
                    'chr12':3,
                    'chr13':4,
                    'chr14':5,
                    'chr15':6,
                    'chr16':7,
                    'chr17':8,
                    'chr18':9,
                    'chr19':10,
                    'chr20':11,
                    'chr2':12,
                    'chr21':13,
                    'chr22':14,
                    'chr3':15,
                    'chr4':16,
                    'chr5':17,
                    'chr6':18,
                    'chr7':19,
                    'chr8':20,
                    'chr9':21,
                    'chrX':22,
                    'chrY':23}


        c_chr, c_start = f_dnase_c.readline().split()[:2]
        r_chr, r_start = f_dnase_r.readline().split()[:2]

        for line in f_label:
            tokens = line.split()


        f_label.close()
        f_dnase_c.close()
        f_dnase_r.close()
