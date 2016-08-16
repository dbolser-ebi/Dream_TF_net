from datareader import *


def print_num_instances_for_each_chromosome():
    chr_size = {}
    datareader = DataReader('../data/')
    for idx, instance in enumerate(datareader.generate_cross_celltype('CTCF', ['MCF-7'])):
        (chromosome, start), features, labels = instance
        if chromosome not in chr_size:
            chr_size[chromosome] = 0
        chr_size[chromosome] += 1

    return chr_size


def read_structure_features():
    hg19MGW = Fasta('../data/annotations/hg19.genome.fa.MGW')
    hg19 = Fasta('../data/annotations/hg19.genome.fa')
    print len(hg19MGW['chr10'][500000:500500].split(','))
    print len(hg19['chr10'][500000:500500])
    '''
    with gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:
        for line in fin:
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            print hg19MGW[chromosome][start:end]
            break
    '''

def get_num_bound_lines():
    reader = DataReader('../data/')
    bound = {}
    for transcription_factor in reader.get_tfs():
        with gzip.open(reader.datapath + reader.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
            fin.readline()
            bound_lines = []
            # only the train set
            for line_idx, line in enumerate(fin):
                if 'B' in line:
                    bound_lines.append(line_idx)
            print transcription_factor, "num bound lines", len(bound_lines)
            bound[transcription_factor] = len(bound_lines)
    print bound



if __name__ == '__main__':
    #print print_num_instances_for_each_chromosome()
    #read_structure_features()
    get_num_bound_lines()
