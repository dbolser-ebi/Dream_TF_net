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


if __name__ == '__main__':
    #print print_num_instances_for_each_chromosome()
    read_structure_features()
