from datareader import *
from convnet import *
from naive import *
from performance_metrics import *
import argparse
import time
from wiggleReader import get_wiggle_output, wiggleToBedGraph, split_iter
from sklearn.ensemble import RandomForestClassifier


class Evaluator:

    def __init__(self, datapath):
        self.num_test_instances = 51676736
        self.num_train_instances = 433306*2
        self.num_tf = 33
        self.num_celltypes = 14
        self.num_train_celltypes = 11
        self.datapath = datapath
        self.datareader = DataReader(datapath)

    def print_results(self, y_test, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in xrange(y_pred.size):
            if y_pred[i] >= 0.5 and y_test[i] == 1:
                TP += 1
            elif y_pred[i] >= 0.5 and y_test[i] == 0:
                FP += 1
            elif y_pred[i] < 0.5 and y_test[i] == 1:
                FN += 1
            else:
                TN += 1
        print 'TP', TP, 'FP', FP, 'TN', TN, 'FN', FN
        print 'AUC ROC', auroc(y_test.flatten(), y_pred.flatten())
        print 'AUC PRC', auprc(y_test.flatten(), y_pred.flatten())
        print 'RECALL AT FDR 0.9', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.90)
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        print 'RECALL AT FDR 0.25', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.25)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)
        print 'RECALL AT FDR 0.05', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.05)

    def make_predictions(self, transcription_factor):
        tf_leaderboard = {
            'ARID3A': ['K562'],
            'ATF2': ['K562'],
            'ATF3': ['liver'],
            'ATF7': ['MCF-7'],
            'CEBPB': ['MCF-7'],
            'CREB1': ['MCF-7'],
            'CTCF': ['GM12878'],
            'E2F6': ['K562'],
            'EGR1': ['K562'],
            'EP300': ['MCF-7'],
            'FOXA1': ['MCF-7'],
            'GABPA': ['K562'],
            'GATA3': ['MCF-7'],
            'JUND': ['H1-hESC'],
            'MAFK': ['K562', 'MCF-7'],
            'MAX': ['MCF-7'],
            'MYC': ['HepG2'],
            'REST': ['K562'],
            'RFX5': ['HepG2'],
            'SPI1': ['K562'],
            'SRF': ['MCF-7'],
            'STAT3': ['GM12878'],
            'TAF1': ['HepG2'],
            'TCF12': ['K562'],
            'TCF7L2': ['MCF-7'],
            'TEAD4': ['MCF-7'],
            'YY1': ['K562'],
            'ZNF143': ['k562']
        }
        tf_final = {
            'ATF2': ['HEPG2'],
            'CTCF': ['PC-3', 'induced_pluripotent_stem_cell'],
            'E2F1': ['K562'],
            'EGR1': ['liver'],
            'FOXA1': ['liver'],
            'FOXA2': ['liver'],
            'GABPA': ['liver'],
            'HNF4A': ['liver'],
            'JUND': ['liver'],
            'MAX': ['liver'],
            'NANOG': ['induced_pluripotent_stem_cell'],
            'REST': ['liver'],
            'TAF1': ['liver']
        }

    def run_within_cell_benchmark(self, model):
        start_benchmark_time = time.time()
        print "Running within cell type benchmark"
        for tf in ['CTCF']:
            print tf
            model.set_transcription_factor(tf)
            celltype = self.datareader.get_celltypes_for_tf(tf)[0]
            chromosomes = ['chr7', 'chr6', 'chr5', 'chr4', 'chr3',
                           'chr2', 'chr9', 'chrX', 'chr13', 'chr12',
                           'chr11', 'chr10', 'chr17', 'chr16', 'chr15',
                           'chr14', 'chr20', 'chr22', 'chr19', 'chr18']
            counts = {'chr7': 3151146, 'chr6': 3419293, 'chr5': 3602250, 'chr4': 3812501, 'chr3': 3955446, 'chr2': 4857748,
             'chr9': 2817543, 'chrX': 3097086, 'chr13': 2303355, 'chr12': 2659379, 'chr11': 2672380, 'chr10': 2702470,
             'chr17': 1622319, 'chr16': 1798121, 'chr15': 2049716, 'chr14': 2145762, 'chr20': 1258450, 'chr22': 1025608,
             'chr19': 1165049, 'chr18': 1561114}

        print "Benchmark completed in: ", (time.time() - start_benchmark_time) / 3600, "hours."

    def run_cross_cell_benchmark(self, model):
        '''
        Run cross celltype benchmark for specified model and model_parameters
        :param datapath: path to data directory
        :param model: Model to be used
        :param model_params: Model parameters
        :return:
        '''
        bin_length = 200
        chromosome_ordering = self.datareader.get_chromosome_ordering()

        print "Running cross celltype benchmark"
        for tf in ['CTCF']:

            #--------------- TRAIN

            print tf
            celltypes = self.datareader.get_celltypes_for_tf(tf)
            if len(celltypes) <= 1:
                continue
            # build model for celltype, using other celltypes
            celltypes_train = celltypes[:-1]
            celltype_test = celltypes[-1]

            # convnet
            X_train = np.zeros((self.num_train_instances, 200, 4), dtype=np.float16)
            y_train = np.zeros((self.num_train_instances, len(celltypes_train)), dtype=np.float16)

            # trees
            #X_train = np.zeros((self.num_train_instances, 7), dtype=np.float32)
            #y_train = np.zeros((self.num_train_instances,), dtype=np.float16)

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(tf,
                                           celltypes_train,
                                           [CrossvalOptions.balance_peaks])):
                if idx >= self.num_train_instances:
                    break
                (chromosome, start), sequence, sequence_features, labels = instance
                # convnet
                X_train[idx, :, :] = self.datareader.sequence_to_one_hot(np.array(list(sequence)))
                y_train[idx, :] = labels

                # tree
                #X_train[idx, :] = sequence_features
                #y_train[idx] = np.max(labels.flatten()[:-1])

            model.fit(X_train, y_train)

            print 'TRAINING COMPLETED'
            self.print_results(y_train[:, -1], model.predict(X_train))

            # free up memory
            del X_train
            del y_train

            # --------------- TEST
            print "Running tests"
            test_chromosomes = ['chr20', 'chr22']

            curr_chr = '-1'

            # tree
            #X_test = np.zeros((self.num_test_instances, 7), dtype=np.float16)
            #y_test = np.zeros((self.num_test_instances,), dtype=np.float16)

            dnase_peaks = self.datareader.get_DNAse_peaks([celltype_test])
            d_idx = 0
            (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
            idx = 0

            y_test = None
            X_test = None

            for instance in self.datareader.generate_cross_celltype(tf,
                                                                    [celltype_test]):
                (chromosome, start), sequence, sequence_features, labels = instance
                if curr_chr == '-1' and chromosome in test_chromosomes:
                    curr_chr = chromosome
                    self.num_test_instances = self.datareader.get_num_instances(chromosome)
                    # convnet
                    X_test = np.zeros((self.num_test_instances, 200, 4), dtype=np.float16)
                    y_test = np.zeros((self.num_test_instances, 1), dtype=np.float16)
                    idx = 0

                elif curr_chr != chromosome and chromosome in test_chromosomes:
                    print 'Results for test', chromosome
                    print 'num test instances', self.num_test_instances
                    y_pred = model.predict(X_test)
                    self.print_results(y_test, y_pred)

                    curr_chr = chromosome
                    self.num_test_instances = self.datareader.get_num_instances(chromosome)
                    # convnet
                    X_test = np.zeros((self.num_test_instances, 200, 4), dtype=np.float16)
                    y_test = np.zeros((self.num_test_instances, 1), dtype=np.float16)
                    idx = 0

                if curr_chr == chromosome:
                    y_test[idx, :] = labels

                    # convnet
                    X_test[idx, :, :] = self.datareader.sequence_to_one_hot(sequence)
                    # X_test[idx] = sequence_features
                    idx += 1
                '''
                # compute if dnase window needs to move
                while chromosome_ordering[chr_dnase] < chromosome_ordering[chromosome]\
                        and d_idx < len(dnase_peaks)-1:
                    d_idx += 1
                    (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
                while chr_dnase == chromosome and start_dnase + bin_length < start \
                        and d_idx < len(dnase_peaks)-1:
                    d_idx += 1

                    (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]

                accessible = chr_dnase == chromosome and (start <= start_dnase+bin_length and start_dnase <= start+bin_length)

                if accessible:
                    X_pred = np.zeros((1, 200, 4), dtype=np.float16)
                    X_pred[0, :, :] = self.datareader.sequence_to_one_hot(features)
                    prediction = model.predict(X_pred)
                    print prediction
                    y_pred[idx, :] = prediction
                else:
                    y_pred[idx, :] = 0
                '''

    def run_multi_benchmark(self, model):
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', help='set tf motif threshold', required=False)
    args = parser.parse_args()

    model = ConvNet('../log/', num_epochs=1, batch_size=512)
    #model = RandomForestClassifier(n_estimators=10)

    evaluator = Evaluator('../data/')
    evaluator.run_cross_cell_benchmark(model)





