from datareader import *
from performance_metrics import *
import argparse
from convnet import *
import sqlite3


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

    def write_results_to_database(self, y_test, y_pred, model, arguments, transcription_factor, dbname='results.db'):
        conn = sqlite3.connect('../results/'+dbname)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS results
        (model TEXT, config TEXT, TF TEXT, TP INTEGER, FP INTEGER, TN INTEGER, FN INTEGER,
        auroc REAL, auprc REAL, rFDR5 REAL, rFDR1 REAL);
        ''')
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

        aroc = auroc(y_test.flatten(), y_pred.flatten())
        aprc = auprc(y_test.flatten(), y_pred.flatten())
        rfdr5 = recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        rfdr1 = recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)

        c.execute("INSERT INTO results VALUES ('%s', '%s', '%s', %d, %d, %d, %d, %f, %f, %f, %f);" %
                  (model, arguments, transcription_factor, TP, FP, TN, FN, aroc, aprc, rfdr5, rfdr1))
        conn.commit()
        conn.close()

    def get_X(self, model, num_instances, bin_size):
        if isinstance(model, ConvNet):
            X = np.zeros((num_instances, bin_size, 4), dtype=np.float32)
        else:
            X = np.zeros((num_instances, 4, bin_size), dtype=np.float32)
        return X

    def get_S(self, model, num_instances, bin_size):
        if isinstance(model, ConvNet):
            S = np.zeros((num_instances, bin_size, 4), dtype=np.float32)
        else:
            S = np.zeros((num_instances, 4, bin_size), dtype=np.float32)
        return S

    def get_y_train(self, model, num_instances, num_celltypes):
        if isinstance(model, ConvNet):
            y_train = np.zeros((num_instances, num_celltypes), dtype=np.float16)
        else:
            y_train = np.zeros((num_instances,), dtype=np.int32)
        return y_train

    def get_da_train(self, model, num_instances, num_celltypes):
        if isinstance(model, ConvNet):
            da_train = np.zeros((num_instances, num_celltypes), dtype=np.float32)
        else:
            da_train = None
        return da_train

    def get_y_test(self, model, num_instances):
        if isinstance(model, ConvNet):
            y_test = np.zeros((num_instances,), dtype=np.float16)
        else:
            y_test = np.zeros((num_instances,), dtype=np.int32)
        return y_test

    def get_da_test(self, model, num_instances):
        if isinstance(model, ConvNet):
            da_test = np.zeros((num_instances,1), dtype=np.float16)
        else:
            da_test = None
        return da_test

    def save_model(self, transcription_factor, model, X_train, S_train, da_train, da_train_val, y_train, y_train_val, run_id):
        np.save(os.path.join(self.datapath, run_id+'Xtrain.npy'), X_train)
        np.save(os.path.join(self.datapath, run_id + 'Strain.npy'),
                S_train)
        np.save(os.path.join(self.datapath, run_id +'datrain.npy'),
                da_train)
        np.save(
            os.path.join(self.datapath, run_id + 'datrain_val.npy'),
            da_train_val)
        np.save(os.path.join(self.datapath, run_id + 'ytrain.npy'), y_train)
        np.save(os.path.join(self.datapath, run_id + 'ytrain_val.npy'), y_train_val)

    def load_model(self, transcription_factor, model, run_id):
        print "Loading train set from file"
        X_train = np.load(os.path.join(self.datapath, run_id + 'Xtrain.npy'))
        S_train = np.load(
            os.path.join(self.datapath, run_id + 'Strain.npy'))
        da_train = np.load(
            os.path.join(self.datapath, run_id + 'datrain.npy'))
        da_train_val = np.load(
            os.path.join(self.datapath, run_id + 'datrain_val.npy'))
        y_train = np.load(os.path.join(self.datapath, run_id + 'ytrain.npy'))
        y_train_val = np.load(os.path.join(self.datapath, run_id + 'ytrain_val.npy'))
        return X_train, S_train, da_train, da_train_val, y_train, y_train_val

    def get_X_data(self, model, sequence):
        if isinstance(model, ConvNet):
            return self.datareader.sequence_to_one_hot(np.array(list(sequence)))
        else:
            return self.datareader.sequence_to_one_hot_transpose(np.array(list(sequence)))

    def get_S_data(self, model, shape_features, bin_size):
        MGW = np.reshape(np.array(shape_features, dtype=np.float32), (bin_size, 4))
        if isinstance(model, ConvNet):
            return MGW
        else:
            return MGW.transpose()

    def get_y_train_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels[:, :-1]
        else:
            return np.max(labels.flatten()[:-1])

    def get_da_train_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels[:, :-1]
        else:
            return None

    def get_y_test_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels
        else:
            return np.max(labels)

    def get_da_test_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels
        else:
            return None

    def make_ladder_predictions(self, model, transcription_factor, unbound_fraction=1, leaderboard=True,
                                ambiguous_as_bound=False, bin_size=200):
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

        tf_mapper = tf_leaderboard if leaderboard else tf_final

        model.set_transcription_factor(transcription_factor)

        for test_celltype in tf_mapper[transcription_factor]:
            self.num_train_instances = int(self.datareader.get_num_bound_lines(transcription_factor) * (1+unbound_fraction))
            print "num train instances", self.num_train_instances
            celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)
            gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes)

            # Run training on full set
            X_train = self.get_X(model, self.num_train_instances, bin_size)
            S_train = self.get_S(model, self.num_train_instances, bin_size)
            da_train = self.get_da_train(model, self.num_train_instances, len(celltypes))
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes))

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(transcription_factor,
                                                                                   celltypes,
                                                                                   [CrossvalOptions.balance_peaks],
                                                                                   unbound_fraction=unbound_fraction,
                                                                                   ambiguous_as_bound=ambiguous_as_bound,
                                                                                   bin_size=bin_size)):
                (_, _), sequence, shape_features, dnase_labels, labels = instance
                X_train[idx] = self.get_X_data(model, sequence)
                y_train[idx] = self.get_y_test_data(model, labels)
                S_train[idx] = self.get_S_data(model, shape_features)
                da_train[idx] = self.get_da_test_data(model, dnase_labels)

            model.fit(X_train, y_train, S_train, gene_expression_features, da_train)

            del X_train
            del S_train
            del y_train
            del da_train

            # Make predictions
            with gzip.open(os.path.join(self.datapath, 'annotations/ladder_regions.blacklistfiltered.bed.gz')) as fin:
                hg19 = Fasta(os.path.join(self.datapath, 'annotations/hg19.genome.fa'))

                test_batch_size = 1000000
                num_test_lines = 8843011
                idx = 0
                X_test = None
                y_test = None
                S_test = None
                da_test = None
                gene_expression_features = self.datareader.get_gene_expression_tpm(test_celltype)
                curr_chromosome = 'chr1'
                shape_features = self.datareader.get_shape_features(curr_chromosome)
                dnase_list = self.datareader.get_DNAse_conservative_peak_lists([test_celltype])

                for l_idx, line in enumerate(fin):
                    if idx == 0:
                        X_test = self.get_X(model, min(test_batch_size, num_test_lines-l_idx), bin_size)
                        S_test = self.get_S(model, min(test_batch_size, num_test_lines-l_idx), bin_size)
                        da_test = self.get_da_test(model, min(test_batch_size, num_test_lines-l_idx))
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])

                    # find position in dnase on the left in sorted order
                    dnase_labels = np.zeros((1, 1), dtype=np.float32)
                    dnase_pos = bisect.bisect_left(dnase_list[0], (chromosome, start, start + 200))
                    # check left
                    if dnase_pos < len(dnase_list[0]):
                        dnase_chr, dnase_start, dnase_end = dnase_list[0][dnase_pos]
                        if dnase_start <= start + 200 and start <= dnase_end:
                            dnase_labels[:, 0] = 1
                    # check right
                    if dnase_pos + 1 < len(dnase_list[0]):
                        dnase_chr, dnase_start, dnase_end = dnase_list[0][dnase_pos + 1]
                        if dnase_start <= start + 200 and start <= dnase_end:
                            dnase_labels[:, 0] = 1

                    if chromosome != curr_chromosome:
                        curr_chromosome = chromosome
                        shape_features = self.datareader.get_shape_features(chromosome)

                    sequence = hg19[chromosome][start:end]
                    X_test[idx] = self.get_X_data(model, sequence)
                    S_test[idx] = self.get_S_data(model, shape_features[start:end])
                    da_test[idx] = dnase_labels

                    idx += 1
                    if idx >= X_test.shape[0]:
                        idx = 0
                        prediction = model.predict(X_test, S_test, gene_expression_features, da_test)
                        if y_test is None:
                            y_test = prediction
                        else:
                            y_test = np.hstack((y_test, prediction))
                fin.seek(0)
                if leaderboard:
                    f_out_name = '../results/'+'L.'+transcription_factor+'.'+test_celltype+'.tab.gz'
                else:
                    f_out_name = '../results/' + 'F.' + transcription_factor + '.' + test_celltype + '.tab.gz'

                with gzip.open(f_out_name, 'w') as fout:
                    for idx, line in enumerate(fin):
                        print>>fout, str(line.strip())+'\t'+str(y_test[idx])

    def run_cross_cell_benchmark(self, model, transcription_factor,
                                 save_train_set=False, unbound_fraction=1,
                                 arguments="", ambiguous_as_bound=False,
                                 bin_size=200):

        print "Running cross celltype benchmark for transcription factor %s" % transcription_factor
        #--------------- TRAIN
        celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)
        self.num_train_instances = int(self.datareader.get_num_bound_lines(transcription_factor)*(1+unbound_fraction))

        model.set_transcription_factor(transcription_factor)

        celltypes_train = celltypes[:-1]
        celltypes_test = celltypes[-1]

        # id for binary data
        run_id = 'models/' + transcription_factor + model.__class__.__name__ + \
                 str(unbound_fraction) + str(ambiguous_as_bound) + str(bin_size)

        if run_id in [mname for mname in os.listdir(os.path.join(self.datapath, 'models/'))]:
            X_train, S_train, da_train, da_train_val, y_train, y_train_val = self.load_model(transcription_factor, model, run_id)
        else:
            y_train_val = np.zeros((self.num_train_instances,), dtype=np.int32)
            X_train = self.get_X(model, self.num_train_instances, bin_size)
            S_train = self.get_S(model, self.num_train_instances, bin_size)
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes_train))
            da_train = self.get_da_train(model, self.num_train_instances, len(celltypes_train))
            da_train_val = np.zeros((self.num_train_instances, 1), dtype=np.float32)

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(transcription_factor,
                                           celltypes,
                                           [CrossvalOptions.balance_peaks],
                                            unbound_fraction=unbound_fraction,
                                            ambiguous_as_bound=ambiguous_as_bound,
                                            bin_size=bin_size)):
                (_, _), sequence, shape_features, dnase_labels, labels = instance
                X_train[idx] = self.get_X_data(model, sequence)
                S_train[idx] = self.get_S_data(model, shape_features, bin_size)
                da_train[idx] = self.get_da_train_data(model, dnase_labels)
                da_train_val[idx] = dnase_labels[:, -1]
                y_train[idx] = self.get_y_train_data(model, labels)
                y_train_val[idx] = labels.flatten()[-1]

        if save_train_set:
            self.save_model(transcription_factor, model, X_train, S_train,
                            da_train, da_train_val, y_train, y_train_val, run_id)

        gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes_train)
        model.fit(X_train, y_train, S_train, gene_expression_features, da_train)

        gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes_test)
        predictions = model.predict(X_train, S_train, gene_expression_features, np.zeros((da_train.shape[0], 1), dtype=np.float32))

        print 'TRAINING COMPLETED'
        self.print_results(y_train_val, predictions)

        # free up memory
        del X_train
        del y_train
        del S_train
        del y_train_val
        del da_train

        # --------------- VALIDATION
        print
        print "RUNNING TESTS"
        test_chromosomes = sorted(['chr10', 'chr11', 'chr12', 'chr13'])
        curr_chr = '-1'

        y_test = None
        X_test = None
        S_test = None
        da_test = None
        idx = 0

        tot_num_test_instances = reduce(lambda x, y: self.datareader.get_num_instances(y)+x, test_chromosomes, 0)
        if isinstance(model, ConvNet):
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.float32)
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.float32)
        else:
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.int32)
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.int32)
        t_idx = 0

        for instance in self.datareader.generate_cross_celltype(transcription_factor,
                                                                [celltypes_test], bin_size=400):

            (chromosome, start), sequence, shape_features, dnase_labels, label = instance
            if test_chromosomes[-1] < chromosome:
                break

            if curr_chr == '-1' and chromosome in test_chromosomes:
                curr_chr = chromosome
                self.num_test_instances = self.datareader.get_num_instances(chromosome)
                X_test = self.get_X(model, self.num_test_instances, bin_size)
                S_test = self.get_S(model, self.num_test_instances, bin_size)
                y_test = self.get_y_test(model, self.num_test_instances)
                da_test = self.get_da_test(model, self.num_test_instances)
                idx = 0

            elif curr_chr != chromosome and chromosome in test_chromosomes:
                print 'Results for test', curr_chr
                print 'num test instances', self.num_test_instances
                y_pred = model.predict(X_test, S_test, gene_expression_features, da_test)
                self.print_results(y_test, y_pred)
                y_tot_test[t_idx:t_idx + self.num_test_instances] = y_test
                y_tot_pred[t_idx:t_idx + self.num_test_instances] = y_pred
                t_idx += self.num_test_instances

                curr_chr = chromosome
                self.num_test_instances = self.datareader.get_num_instances(chromosome)
                X_test = self.get_X(model, self.num_test_instances, bin_size)
                S_test = self.get_S(model, self.num_test_instances, bin_size)
                y_test = self.get_y_test(model, self.num_test_instances)
                da_test = self.get_da_test(model, self.num_test_instances)
                idx = 0

            elif curr_chr != '-1' and curr_chr != chromosome:
                print 'Results for test', curr_chr
                print 'num test instances', self.num_test_instances
                y_pred = model.predict(X_test, S_test, gene_expression_features, da_test)

                self.print_results(y_test, y_pred)
                y_tot_test[t_idx:t_idx+self.num_test_instances] = y_test
                y_tot_pred[t_idx:t_idx+self.num_test_instances] = y_pred
                t_idx += self.num_test_instances

                curr_chr = '-1'

            if curr_chr == chromosome:
                y_test[idx] = label
                X_test[idx] = self.get_X_data(model, sequence)
                S_test[idx] = self.get_S_data(model, shape_features, bin_size)
                da_test[idx] = dnase_labels
                idx += 1

        print "Overall test results"
        self.print_results(y_tot_test, y_tot_pred)
        self.write_results_to_database(y_tot_test, y_tot_pred, model.__class__.__name__,
                                       arguments, transcription_factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription_factors', '-tfs', help='Comma separated list of transcription factors', required=True)
    parser.add_argument('--model', '-m', help='Choose model [TFC/THC]', required=True)
    parser.add_argument('--validate', '-v', action='store_true', help='run cross TF validation benchmark', required=False)
    parser.add_argument('--ladder', '-l', action='store_true', help='predict TF ladderboard', required=False)
    parser.add_argument('--test', '-t', action='store_true', help='predict TF final round', required=False)
    parser.add_argument('--config', '-c', help='configuration of model', required=False)
    parser.add_argument('--unbound_fraction', '-uf', help='unbound fraction in training', required=False)
    parser.add_argument('--num_epochs', '-ne', help='number of epochs', required=False)
    parser.add_argument('--ambiguous_bound', '-ab', action='store_true', help='treat ambiguous as bound', required=False)
    parser.add_argument('--bin_size', '-bs', help='Sequence bin size (must be an even number >= 200)', required=False)
    args = parser.parse_args()
    model = None

    num_epochs = 1 if args.num_epochs is None else int(args.num_epochs)
    config = int(1 if args.config is None else args.config)
    bin_size = int(200 if args.bin_size is None else args.bin_size)
    bin_size -= bin_size % 2

    if args.model == 'TFC':
        model = ConvNet('../log/', num_epochs=num_epochs, batch_size=512,
                        num_gen_expr_features=32, config=config, dropout_rate=0.25,
                        eval_size=0.2, num_shape_features=4, sequence_width=bin_size)
    elif args.model == 'THC':
        from theanonet import *
        model = DNNClassifier(200, 4, 0.2, [100], [0.5], verbose=True, max_epochs=100, batch_size=512)

    else:
        print "Model options: TFC (TensorFlow Convnet), THC (Theano Convnet)"

    unbound_fraction = 1

    if args.unbound_fraction is not None:
        unbound_fraction = float(args.unbound_fraction)

    if model is not None:
        transcription_factors = args.transcription_factors.split(',')
        evaluator = Evaluator('../data/')
        for transcription_factor in transcription_factors:
            if args.validate:
                evaluator.run_cross_cell_benchmark(model, transcription_factor, save_train_set=True,
                                                   unbound_fraction=unbound_fraction,
                                                   arguments=str(vars(args)).replace('\'', ''),
                                                   ambiguous_as_bound=args.ambiguous_bound,
                                                   bin_size=bin_size)
            if args.ladder:
                evaluator.make_ladder_predictions(model, transcription_factor, unbound_fraction=unbound_fraction, leaderboard=True)

            if args.test:
                evaluator.make_ladder_predictions(model, transcription_factor, unbound_fraction, False)
