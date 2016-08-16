from datareader import *
from performance_metrics import *
import argparse
from convnet import *
from sklearn.ensemble import RandomForestClassifier


class Evaluator:

    def __init__(self, datapath):
        self.num_test_instances = 51676736
        self.num_tf = 33
        self.num_celltypes = 14
        self.num_train_celltypes = 11
        self.datapath = datapath
        self.datareader = DataReader(datapath)
        self.num_train_instances = 1000

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
        print 'AU ROC', auroc(y_test.flatten(), y_pred.flatten())
        print 'AU PRC', auprc(y_test.flatten(), y_pred.flatten())
        print 'RECALL AT FDR 0.9', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.90)
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        print 'RECALL AT FDR 0.25', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.25)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)
        print 'RECALL AT FDR 0.05', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.05)

    def get_X(self, model, num_instances):
        if isinstance(model, ConvNet):
            X = np.zeros((num_instances, 200, 4), dtype=np.float16)
        elif isinstance(model, RandomForestClassifier):
            num_sequence_features = self.datareader.get_num_sequence_features(tf)
            X = np.zeros((num_instances, num_sequence_features), dtype=np.float32)
        else:
            X = np.zeros((num_instances, 4, 200), dtype=np.float32)
        return X

    def get_y_train(self, model, num_instances, num_celltypes):
        if isinstance(model, ConvNet):
            y_train = np.zeros((num_instances, num_celltypes), dtype=np.float16)
        elif isinstance(model, RandomForestClassifier):
            y_train = np.zeros((num_instances,), dtype=np.int32)
        else:
            y_train = np.zeros((num_instances,), dtype=np.int32)
        return y_train

    def get_y_test(self, model, num_instances):
        if isinstance(model, ConvNet):
            y_test = np.zeros((num_instances,), dtype=np.float16)
        elif isinstance(model, RandomForestClassifier):
            y_test = np.zeros((num_instances,), dtype=np.int32)
        else:
            y_test = np.zeros((num_instances,), dtype=np.int32)
        return y_test

    def save_model(self, model, X_train, y_train, y_train_val):
        np.save(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'Xtrain.npy'), X_train)
        np.save(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'ytrain.npy'), y_train)
        np.save(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'ytrain_val.npy'), y_train_val)

    def load_model(self, model):
        print "Loading train set from file"
        X_train = np.load(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'Xtrain.npy'))
        y_train = np.load(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'ytrain.npy'))
        y_train_val = np.load(os.path.join(self.datapath, 'models/' + model.__class__.__name__ + 'ytrain_val.npy'))
        return X_train, y_train, y_train_val

    def get_X_data(self, model, sequence, sequence_features=None):
        if isinstance(model, ConvNet):
            return self.datareader.sequence_to_one_hot(np.array(list(sequence)))
        elif isinstance(model, RandomForestClassifier):
            return sequence_features
        else:
            return self.datareader.sequence_to_one_hot_transpose(np.array(list(sequence)))

    def get_y_train_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels[:, :-1]
        else:
            return np.max(labels.flatten()[:-1])

    def get_y_test_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels
        else:
            return np.max(labels)


    def make_test_predictions(self):
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

    def make_ladder_predictions(self, model, transcription_factor):
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
        for test_celltype in tf_leaderboard[transcription_factor]:
            self.num_train_instances = self.datareader.get_num_bound_lines(transcription_factor) * 2
            celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)
            # Run training on full set
            X_train = self.get_X(model, self.num_train_instances)
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes))

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(transcription_factor,
                                                                                   celltypes,
                                                                                   [CrossvalOptions.balance_peaks])):
                if idx >= self.num_train_instances:
                    break
                (_, _), sequence, sequence_features, labels = instance
                X_train[idx] = self.get_X_data(model, sequence, sequence_features)
                y_train[idx] = self.get_y_test_data(model, labels)
            model.fit(X_train, y_train)

            # Make predictions
            with gzip.open(os.path.join(self.datapath, 'annotations/ladder_regions.blacklistfiltered.bed.gz')) as fin:
                hg19 = Fasta(os.path.join(self.datapath, 'annotations/hg19.genome.fa'))

                test_batch_size = 1000000
                num_test_lines = 8843011
                idx = 0
                X_test = None
                y_test = None

                for l_idx, line in enumerate(fin):
                    if idx == 0:
                        X_test = self.get_X(model, min(test_batch_size, num_test_lines-l_idx))
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])
                    sequence = hg19[chromosome][start:end]
                    X_test[idx] = self.get_X_data(model, sequence)
                    idx += 1
                    if idx >= X_test.shape[0]:
                        idx = 0
                        prediction = model.predict(X_test)
                        if y_test is None:
                            y_test = prediction
                        else:
                            y_test = np.hstack((y_test, prediction))
                fin.seek(0)
                with gzip.open('../results/'+'L.'+transcription_factor+'.'+test_celltype+'.tab.gz', 'w') as fout:
                    for idx, line in enumerate(fin):
                        print>>fout, str(line.strip())+'\t'+str(y_test[idx])

    def run_cross_cell_benchmark(self, model, transcription_factor, save_train_set=False):
        '''
        Run cross celltype benchmark for specified model and model_parameters
        :param datapath: path to data directory
        :param model: Model to be used
        :return:
        '''
        print "Running cross celltype benchmark for transcription factor %s" % transcription_factor
        #--------------- TRAIN

        print transcription_factor
        celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)

        gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes)
        self.num_train_instances = self.datareader.get_num_bound_lines(transcription_factor)*2
        if len(celltypes) <= 1:
            print 'cant build cross transcription_factor validation for this transcription factor'
            return

        celltypes_train = celltypes[:-1]
        celltype_test = celltypes[-1]

        if os.path.exists(os.path.join(self.datapath, 'models/'+model.__class__.__name__+'Xtrain.npy')):
            X_train, y_train, y_train_val = self.load_model(model)
        else:
            y_train_val = np.zeros((self.num_train_instances,), dtype=np.int32)
            X_train = self.get_X(model, self.num_train_instances)
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes_train))

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(transcription_factor,
                                           celltypes,
                                           [CrossvalOptions.balance_peaks])):
                if idx >= self.num_train_instances:
                    break
                (_, _), sequence, sequence_features, labels = instance
                X_train[idx] = self.get_X_data(model, sequence, sequence_features)
                y_train[idx] = self.get_y_train_data(model, labels)
                y_train_val[idx] = labels.flatten()[-1]

        if save_train_set:
            self.save_model(model, X_train, y_train, y_train_val)

        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        print 'TRAINING COMPLETED'
        self.print_results(y_train_val, predictions)

        # free up memory
        del X_train
        del y_train

        # --------------- VALIDATION
        print
        print "RUNNING TESTS"
        test_chromosomes = ['chr10', 'chr4', 'chr20', 'chr22']#self.datareader.get_chromosomes()
        curr_chr = '-1'

        y_test = None
        X_test = None
        idx = 0

        tot_num_test_instances = reduce(lambda x, y: self.datareader.get_num_instances(y)+x, test_chromosomes, 0)
        if isinstance(model, ConvNet):
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.float16)
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.float16)
        else:
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.int32)
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.int32)
        t_idx = 0

        for instance in self.datareader.generate_cross_celltype(transcription_factor,
                                                                [celltype_test]):
            (chromosome, start), sequence, sequence_features, label = instance
            if curr_chr == '-1' and chromosome in test_chromosomes:
                curr_chr = chromosome
                self.num_test_instances = self.datareader.get_num_instances(chromosome)
                X_test = self.get_X(model, self.num_test_instances)
                y_test = self.get_y_test(model, self.num_test_instances)
                idx = 0

            elif curr_chr != chromosome and chromosome in test_chromosomes:
                print 'Results for test', curr_chr
                print 'num test instances', self.num_test_instances
                y_pred = model.predict(X_test)
                self.print_results(y_test, y_pred)
                y_tot_test[t_idx:t_idx + self.num_test_instances] = y_test
                y_tot_pred[t_idx:t_idx + self.num_test_instances] = y_pred
                t_idx += self.num_test_instances

                curr_chr = chromosome
                self.num_test_instances = self.datareader.get_num_instances(chromosome)
                X_test = self.get_X(model, self.num_test_instances)
                y_test = self.get_y_test(model, self.num_test_instances)
                idx = 0

            elif curr_chr != '-1' and curr_chr != chromosome:
                print 'Results for test', curr_chr
                print 'num test instances', self.num_test_instances
                y_pred = model.predict(X_test)

                self.print_results(y_test, y_pred)
                y_tot_test[t_idx:t_idx+self.num_test_instances] = y_test
                y_tot_pred[t_idx:t_idx+self.num_test_instances] = y_pred
                t_idx += self.num_test_instances

                curr_chr = '-1'

            if curr_chr == chromosome:
                y_test[idx] = label
                X_test[idx] = self.get_X_data(model, sequence, sequence_features)
                idx += 1

        print "Overall test results"
        self.print_results(y_tot_test, y_tot_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription_factor', '-tf', help='Choose transcription factor', required=True)
    parser.add_argument('--model', help='Choose model [TFC/THC/RF]', required=True)
    parser.add_argument('--validate', '-v', action='store_true', help='run cross TF validation benchmark')
    parser.add_argument('--predict', '-p', action='store_true', help='predict TF ladderboard')
    args = parser.parse_args()

    model = None

    if args.model == 'TFC':
        model = ConvNet('../log/', num_epochs=1, batch_size=512)
    elif args.model == 'RF':
        model = RandomForestClassifier(n_estimators=333, max_features="sqrt")
    elif args.model == 'THC':
        from theanonet import *
        model = DNNClassifier(200, 4, 0.2, [100], [0.5], verbose=True, max_epochs=100, batch_size=512)

    else:
        print "Model options: TFC (TensorFlow Convnet), THC (Theano Convnet), RF (Random Forest)"

    if model is not None:
        transcription_factor = args.transcription_factor
        evaluator = Evaluator('../data/')
        if args.validate:
            evaluator.run_cross_cell_benchmark(model, transcription_factor, save_train_set=True)
        if args.predict:
            evaluator.make_ladder_predictions(model, transcription_factor)
