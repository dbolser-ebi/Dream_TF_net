from datareader import *
from convnet import *
from naive import *
from performance_metrics import *
import argparse
import time
from wiggleReader import get_wiggle_output, wiggleToBedGraph, split_iter


class Evaluator:

    def __init__(self, datapath):
        self.num_test_instances = 51676736
        self.num_train_instances = 100000
        self.num_tf = 33
        self.num_celltypes = 14
        self.num_train_celltypes = 11
        self.datapath = datapath
        self.datareader = DataReader(datapath)

    def make_predictions(self):
        return

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
        start_benchmark_time = time.time()

        print "Running cross celltype benchmark"
        for tf in ['CTCF']:

            #--------------- TRAIN

            print tf
            celltypes = self.datareader.get_celltypes_for_tf(tf)
            if len(celltypes) <= 1:
                # cant build model for this TF
                continue
            # build model for celltype, using other celltypes
            celltypes_train = celltypes[:-1]
            celltype_test = celltypes[-1]

            X_train = np.zeros((self.num_train_instances, 200, 4), dtype=np.float16)
            y_train = np.zeros((self.num_train_instances, len(celltypes_train)), dtype=np.float16)

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(tf,
                                           celltypes_train,
                                           [CrossvalOptions.filter_on_DNase_peaks])):
                if idx >= self.num_train_instances:
                    break
                sequence, labels = instance
                X_train[idx, :, :] = self.datareader.sequence_to_one_hot(np.array(list(sequence)))
                y_train[idx, :] = labels

            model.fit(X_train, y_train)

            print 'TRAINING COMPLETED: ', int(time.time()-start_benchmark_time), 's'

            # free up memory
            del X_train
            del y_train

            # --------------- TEST

            test_chromosomes = ['chr20']
            self.num_test_instances = self.datareader.get_num_instances(test_chromosomes[0])
            y_test = np.zeros((self.num_test_instances, 1), dtype=np.float16)
            y_pred = np.zeros((self.num_test_instances, 1), dtype=np.float16)
            X_test = np.zeros((self.num_test_instances, 200, 4), dtype=np.float16)
            dnase_peaks = self.datareader.get_DNAse_peaks([celltype_test])
            d_idx = 0
            (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
            chromosome_ordering = self.datareader.get_chromosome_ordering()
            bin_length = 200
            idx = 0

            for instance in self.datareader.generate_cross_celltype(tf,
                                           [celltype_test]):
                if idx >= self.num_test_instances:
                    break
                (chromosome, start), sequence, labels = instance
                if chromosome not in test_chromosomes:
                    continue


                # compute if dnase window needs to move
                while chromosome_ordering[chr_dnase] < chromosome_ordering[chromosome]\
                        and d_idx < len(dnase_peaks)-1:
                    d_idx += 1
                    (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
                while chr_dnase == chromosome and start_dnase + bin_length < start \
                        and d_idx < len(dnase_peaks)-1:
                    d_idx += 1

                    (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
                '''
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
                y_test[idx, :] = labels
                X_test[idx, :, :] = self.datareader.sequence_to_one_hot(sequence)
                idx += 1
            y_pred = model.predict(X_test)

            print 'AUC ROC', auroc(y_test.flatten(), y_pred.flatten())
            print 'AUC PRC', auprc(y_test.flatten(), y_pred.flatten())
            print 'RECALL AT FDR 0.9', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.90)
            print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
            print 'RECALL AT FDR 0.25', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.25)
            print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)
            print 'RECALL AT FDR 0.05', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.05)
            print 'TIME: ', int(time.time()-start_benchmark_time), 's'

        print "Benchmark completed in: ", (time.time() - start_benchmark_time)/3600, "hours."

    def run_multi_benchmark(self, model):
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', help='set tf motif threshold', required=False)
    args = parser.parse_args()
    #model = TFMotifScanner(threshold=1.0 if args.threshold is None else float(args.threshold))
    model = ConvNet('../log/')
    evaluator = Evaluator('../data/')
    evaluator.run_cross_cell_benchmark(model)





