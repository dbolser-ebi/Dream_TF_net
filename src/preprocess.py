from datareader import *
from wiggleReader import *
import threading
from multiprocessing import Process

def print_num_instances_for_each_chromosome():
    chr_size = {}
    datareader = DataReader('../data/')
    for idx, instance in enumerate(datareader.generate_cross_celltype('CTCF', ['MCF-7'])):
        (chromosome, start), features, labels = instance
        if chromosome not in chr_size:
            chr_size[chromosome] = 0
        chr_size[chromosome] += 1

    return chr_size


def get_num_bound_lines(amb_as_bound=False):
    reader = DataReader('../data/')
    bound = {}
    for transcription_factor in reader.get_tfs():
        with gzip.open(reader.datapath + reader.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
            fin.readline()
            bound_lines = []
            # only the train set
            for line_idx, line in enumerate(fin):
                if 'B' in line or (amb_as_bound and 'A' in line):
                    bound_lines.append(line_idx)
            print transcription_factor, "num bound lines", len(bound_lines)
            bound[transcription_factor] = len(bound_lines)
    print bound


def dnase_fold(celltype, left, right):
    fpath = os.path.join('../data/', 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
    with open(fpath) as fin:
        fin.read()
    process = Popen(["wiggletools", "seek", "chr1", str(left), str(right),  fpath],
                    stdout=PIPE)
    (wiggle_output, _) = process.communicate()
    track = np.zeros((right-left+1, ), dtype=np.float32)
    start = left
    for line in split_iter(wiggle_output):
        tokens = line.split()
        if line.startswith('fixedStep'):
            start = int(tokens[2].split('=')[1])-left
        elif line.startswith('chr'):
            chromosome = tokens[0]
            start = int(tokens[1])+1-left
            end = int(tokens[2])
            value = float(tokens[3])
            track[start:end] = value
        else:
            value = float(tokens[0])
            track[start] = value
            start += 1
    return track


def get_DNAse_fold_track(celltype, chromosome, left, right):
    fpath = os.path.join('../data/', 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
    with open(fpath) as fin:
        fin.read()
    process = Popen(["wiggletools", "seek", chromosome, str(left), str(right), fpath],
                    stdout=PIPE)
    (wiggle_output, _) = process.communicate()
    track = np.zeros((right - left + 1,), dtype=np.float32)
    position = 0
    for line in split_iter(wiggle_output):
        tokens = line.split()
        if line.startswith('fixedStep'):
            continue
        elif line.startswith('chr'):
            start = int(tokens[1]) + 1
            end = int(tokens[2])
            length = end-start+1
            value = float(tokens[3])
            track[position:position+length] = value
            position += length
        else:
            value = float(tokens[0])
            track[position] = value
            position += 1
    return track


class DNAseSignalProcessor(threading.Thread):
    def __init__(self, lines, fout_path, celltype):
        super(DNAseSignalProcessor, self).__init__()
        self.lines = lines
        self.fout_path = fout_path
        self.celltype = celltype

    def run(self):
        with open(self.fout_path, 'w') as fout:
            for line in split_iter(self.lines):
                print line
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                track = get_DNAse_fold_track(self.celltype, chromosome, start, end)
                for i in range(0, track.size - 200, 50):
                    sbin = track[i:i + 200]
                    print>>fout, np.max(sbin), np.percentile(sbin, 90), np.mean(sbin)


def parralelSignalProcessor(lines, fout_path, celltype):
    with open(fout_path, 'w') as fout:
        for line in split_iter(lines):
            print line
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            track = get_DNAse_fold_track(celltype, chromosome, start, end)
            for i in range(0, track.size - 200, 50):
                sbin = track[i:i + 200]
                print>> fout, np.max(sbin), np.percentile(sbin, 90), np.mean(sbin)


def preprocess_dnase():
    reader = DataReader('../data/')
    num_train_instances = 51676736
    threads = []
    processes = []

    celltypes = reader.get_celltypes()
    print celltypes, len(celltypes)

    with open('../data/annotations/train_regions.blacklistfiltered.merged.bed') as fin:
        lines = fin.read()

    for celltype in celltypes:
        def process_dnase(fin, fout):
            for line in fin:
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                track = get_DNAse_fold_track(celltype, chromosome, start, end)
                for i in range(0, track.size-200, 50):
                    sbin = track[i:i+200]
                    print >>fout, np.max(sbin), np.percentile(sbin, 90), np.mean(sbin)

        if not os.path.exists('../data/preprocess/DNASE_FEATURES/%s_train.txt' % celltype):
            fout_path = '../data/preprocess/DNASE_FEATURES/%s_train.txt' % celltype
            '''
            threads.append(DNAseSignalProcessor(lines,
                                                    ,
                                                    celltype))
            '''
            processes.append(Process(target=parralelSignalProcessor, args=(lines, fout_path, celltype,)))
                    #process_dnase(fin, fout)
        '''
        if not os.path.exists('../data/preprocess/DNASE_FEATURES/%s_ladder.txt' % celltype):
            with open('../data/preprocess/DNASE_FEATURES/%s_ladder.txt' % celltype, 'w') as fout:
                #process_dnase(fin, fout)
                threads.append(DNAseSignalProcessor('../data/annotations/ladder_regions.blacklistfiltered.merged.bed', fout, celltype))

        if not os.path.exists('../data/preprocess/DNASE_FEATURES/%s_test.txt' % celltype):
            with open('../data/preprocess/DNASE_FEATURES/%s_test.txt' % celltype, 'w') as fout:
                #process_dnase(fin, fout)
                threads.append(DNAseSignalProcessor('../data/annotations/test_regions.blacklistfiltered.merged.bed', fout, celltype))
        '''
    '''
    num_simul_threads = 14
    for i in range(0, len(threads), num_simul_threads):
        map(lambda x: x.start(), threads[i:i+num_simul_threads])
        map(lambda x: x.join(), threads[i:i+num_simul_threads])
    '''
    num_processes = 14
    for i in range(0, len(processes), num_processes):
        map(lambda x: x.start(), processes[i:i + num_processes])
        map(lambda x: x.join(), processes[i:i + num_processes])


if __name__ == '__main__':
    #print print_num_instances_for_each_chromosome()
    #read_structure_features()
    #get_num_bound_lines(True)
    #print dnase_fold('A549', 10600, 11000)
    preprocess_dnase()

