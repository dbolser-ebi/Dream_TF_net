from datareader import *
from performance_metrics import *
import argparse
from convnet import *
import sqlite3
from progress.bar import Bar


def finish_bar(bar):
    if bar is not None:
        bar.finish()


class Evaluator:

    def __init__(self, datapath, bin_size=200, ambiguous_as_bound=False, regression=False, show_progress=True, num_dnase_features=4):
        self.num_dnase_features = num_dnase_features
        self.num_train_instances = 51676736
        self.num_tf = 32
        self.num_celltypes = 14
        self.num_train_celltypes = 11
        self.datapath = datapath
        self.datareader = DataReader(datapath)
        self.bin_size = bin_size
        self.ambiguous_as_bound = ambiguous_as_bound
        self.regression = regression
        self.show_progress = show_progress

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

    def get_da_train(self, model, num_instances, num_celltypes, num_features):
        if isinstance(model, ConvNet):
            da_train = np.zeros((num_instances, num_features, num_celltypes), dtype=np.float32)
        else:
            da_train = None
        return da_train

    def get_y_test(self, model, num_instances):
        if isinstance(model, ConvNet):
            y_test = np.zeros((num_instances,), dtype=np.float16)
        else:
            y_test = np.zeros((num_instances,), dtype=np.int32)
        return y_test

    def get_da_test(self, model, num_instances, num_features):
        if isinstance(model, ConvNet):
            da_test = np.zeros((num_instances, num_features, 1), dtype=np.float16)
        else:
            da_test = None
        return da_test

    def save_model(self, X_train, S_train, da_train, da_train_val, y_train, y_train_val, run_id):
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

    def load_model(self, run_id):
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

    def get_da_train_data(self, model, features):
        if isinstance(model, ConvNet):
            return features[:, :-1]
        else:
            return None

    def get_y_test_data(self, model, labels):
        if isinstance(model, ConvNet):
            return labels
        else:
            return np.max(labels)

    def get_da_test_data(self, model, features):
        if isinstance(model, ConvNet):
            return features
        else:
            return None

    def make_ladder_predictions(self, model, transcription_factor, unbound_fraction=1, leaderboard=True):
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

        part = 'ladder' if leaderboard else 'test'

        model.set_transcription_factor(transcription_factor)

        for test_celltype in tf_mapper[transcription_factor]:
            self.num_train_instances = int(self.datareader.get_num_bound_lines(transcription_factor, self.ambiguous_as_bound) * (1+unbound_fraction))
            print "num train instances", self.num_train_instances
            celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)
            gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes)

            # Run training on full set
            X_train = self.get_X(model, self.num_train_instances, self.bin_size)
            S_train = self.get_S(model, self.num_train_instances, self.bin_size)
            da_train = self.get_da_train(model, self.num_train_instances, len(celltypes), self.num_dnase_features)
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes))

            for idx, instance in enumerate(self.datareader.generate_cross_celltype(part, transcription_factor,
                                                                                   celltypes,
                                                                                   self.num_dnase_features,
                                                                                   options=[CrossvalOptions.balance_peaks],
                                                                                   unbound_fraction=unbound_fraction,
                                                                                   ambiguous_as_bound=self.ambiguous_as_bound,
                                                                                   bin_size=self.bin_size,
                                                                                   regression=self.regression)):
                (_, _), sequence, shape_features, dnase_features, labels = instance
                X_train[idx] = self.get_X_data(model, sequence)
                y_train[idx] = self.get_y_test_data(model, labels)
                S_train[idx] = self.get_S_data(model, shape_features, self.bin_size)
                da_train[idx] = self.get_da_test_data(model, dnase_features)

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
                dnase_lists = self.datareader.get_DNAse_conservative_peak_lists([test_celltype])

                dnase_feature_handlers = []
                for celltype in celltypes:
                    f = open(os.path.join(self.datapath, 'preprocess/DNASE_FEATURES/%s_%s_%d.txt' % (celltype, part, bin_size)))
                    dnase_feature_handlers.append(f)

                chipseq_feature_handlers = []
                for celltype in celltypes:
                    f = gzip.open(os.path.join(self.datapath, 'preprocess/CHIPSEQ_FEATURES/_%s_%s_%d' % (
                    celltype, transcription_factor, bin_size)))
                    chipseq_feature_handlers.append(f)

                for l_idx, line in enumerate(fin):
                    dnase_feature_lines = []
                    for handler in dnase_feature_handlers:
                        dnase_feature_lines.append(handler.next())
                    chipseq_feature_lines = []
                    for handler in chipseq_feature_handlers:
                        chipseq_feature_lines.append(handler.next())

                    if idx == 0:
                        X_test = self.get_X(model, min(test_batch_size, num_test_lines-l_idx), self.bin_size)
                        S_test = self.get_S(model, min(test_batch_size, num_test_lines-l_idx), self.bin_size)
                        da_test = self.get_da_test(model, min(test_batch_size, num_test_lines-l_idx), self.num_dnase_features)
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])

                    # find position in dnase on the left in sorted order
                    dnase_labels = np.zeros((1, len(celltypes)), dtype=np.float32)
                    for c_idx, celltype in enumerate(celltypes):
                        dnase_pos = bisect.bisect_left(dnase_lists[c_idx], (chromosome, start, start + 200))
                        # check left
                        if dnase_pos < len(dnase_lists[c_idx]):
                            dnase_chr, dnase_start, dnase_end = dnase_lists[c_idx][dnase_pos]
                            if dnase_start <= start + 200 and start <= dnase_end:
                                dnase_labels[:, c_idx] = 1
                        # check right
                        if dnase_pos + 1 < len(dnase_lists[c_idx]):
                            dnase_chr, dnase_start, dnase_end = dnase_lists[c_idx][dnase_pos + 1]
                            if dnase_start <= start + 200 and start <= dnase_end:
                                dnase_labels[:, c_idx] = 1

                    # dnase fold coverage
                    dnase_fold_coverage = np.zeros((self.num_dnase_features, len(celltypes)), dtype=np.float32)

                    for c_idx, celltype in enumerate(celltypes):
                        tokens = map(float, dnase_feature_lines[c_idx].split())
                        dnase_fold_coverage[0, c_idx] = dnase_labels[0, c_idx]
                        for t_idx, token in enumerate(tokens):
                            dnase_fold_coverage[t_idx+1, c_idx] = token

                    if chromosome != curr_chromosome:
                        curr_chromosome = chromosome
                        shape_features = self.datareader.get_shape_features(chromosome)

                    sequence = hg19[chromosome][start:end]
                    X_test[idx] = self.get_X_data(model, sequence)
                    S_test[idx] = self.get_S_data(model, shape_features[start:end], self.bin_size)
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
                    f_out_name = '../results/' + 'L.'+transcription_factor+'.'+test_celltype+'.tab.gz'
                else:
                    f_out_name = '../results/' + 'F.' + transcription_factor + '.' + test_celltype + '.tab.gz'

                with gzip.open(f_out_name, 'w') as fout:
                    for idx, line in enumerate(fin):
                        print>>fout, str(line.strip())+'\t'+str(y_test[idx])

    def run_cross_cell_benchmark(self, model, transcription_factor,
                                 save_train_set=False, unbound_fraction=1,
                                 arguments=""):

        print "Running cross celltype benchmark for transcription factor %s" % transcription_factor
        #--------------- TRAIN
        celltypes = self.datareader.get_celltypes_for_tf(transcription_factor)
        self.num_train_instances = int(self.datareader.get_num_bound_lines(transcription_factor, self.ambiguous_as_bound)*(1+unbound_fraction))

        model.set_transcription_factor(transcription_factor)

        celltypes_train = celltypes[:-1]
        celltypes_test = celltypes[-1]

        # id for binary data
        run_id = 'models/' + transcription_factor + model.__class__.__name__ + \
                 "_ub_"+str(unbound_fraction) + "_ab_" + str(self.ambiguous_as_bound)\
                 + "bs" + str(self.bin_size) + "_reg_" + str(self.regression)

        def already_computed():
            fnames = [mname for mname in os.listdir(os.path.join(self.datapath, 'models/'))]
            for fname in fnames:
                if run_id[7:] in fname:
                    return True
            return False

        if already_computed():
            X_train, S_train, da_train, da_train_val, y_train, y_train_val = self.load_model(run_id)
        else:
            y_train_val = np.zeros((self.num_train_instances,), dtype=np.int32)
            X_train = self.get_X(model, self.num_train_instances, self.bin_size)
            S_train = self.get_S(model, self.num_train_instances, self.bin_size)
            y_train = self.get_y_train(model, self.num_train_instances, len(celltypes_train))
            da_train = self.get_da_train(model, self.num_train_instances, len(celltypes_train), self.num_dnase_features)
            da_train_val = np.zeros((self.num_train_instances, self.num_dnase_features, 1), dtype=np.float32)

            if self.show_progress:
                bar = Bar('Loading train set', fill='#', suffix='%(percent)d%%', max=self.num_train_instances)

            for idx, instance in enumerate(self.datareader.generate_cross_celltype('train', transcription_factor,
                                           celltypes,
                                            self.num_dnase_features,
                                           options=[CrossvalOptions.balance_peaks],
                                            unbound_fraction=unbound_fraction,
                                            ambiguous_as_bound=self.ambiguous_as_bound,
                                            bin_size=self.bin_size,
                                            regression=self.regression
                                            )):
                (_, _), sequence, shape_features, dnase_features, labels = instance
                X_train[idx] = self.get_X_data(model, sequence)
                S_train[idx] = self.get_S_data(model, shape_features, self.bin_size)
                da_train[idx] = self.get_da_train_data(model, dnase_features)
                da_train_val[idx, :, :] = np.reshape(dnase_features[:, -1], (self.num_dnase_features, 1))
                y_train[idx] = self.get_y_train_data(model, labels)
                y_train_val[idx] = labels.flatten()[-1]
                if self.show_progress:
                    bar.next()

            if self.show_progress:
                bar.finish()

            if save_train_set:
                self.save_model(X_train, S_train,
                                da_train, da_train_val, y_train, y_train_val, run_id)

        gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes_train)
        model.fit(X_train, y_train, S_train, gene_expression_features, da_train)

        gene_expression_features = self.datareader.get_gene_expression_tpm(celltypes_test)
        predictions = model.predict(X_train, S_train, gene_expression_features, da_train_val)

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
        bar = None
        idx = 0

        tot_num_test_instances = reduce(lambda x, y: self.datareader.get_num_instances(y)+x, test_chromosomes, 0)
        if isinstance(model, ConvNet):
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.float32)
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.float32)
        else:
            y_tot_pred = np.zeros((tot_num_test_instances,), dtype=np.int32)
            y_tot_test = np.zeros((tot_num_test_instances,), dtype=np.int32)
        t_idx = 0

        for instance in self.datareader.generate_cross_celltype('train', transcription_factor,
                                                                [celltypes_test],
                                                                num_dnase_features=self.num_dnase_features,
                                                                bin_size=self.bin_size,
                                                                regression=self.regression):

            (chromosome, start), sequence, shape_features, dnase_features, label = instance
            if test_chromosomes[-1] < chromosome and curr_chr == '-1':
                break

            if curr_chr == '-1' and chromosome in test_chromosomes:

                curr_chr = chromosome
                self.num_test_instances = self.datareader.get_num_instances(chromosome)
                X_test = self.get_X(model, self.num_test_instances, self.bin_size)
                S_test = self.get_S(model, self.num_test_instances, self.bin_size)
                y_test = self.get_y_test(model, self.num_test_instances)
                da_test = self.get_da_test(model, self.num_test_instances, self.num_dnase_features)
                if self.show_progress:
                    finish_bar(bar)
                    bar = Bar('Loading chromosome %s' % curr_chr, fill='#', suffix='%(percent)d%%', max=self.num_test_instances)
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
                X_test = self.get_X(model, self.num_test_instances, self.bin_size)
                S_test = self.get_S(model, self.num_test_instances, self.bin_size)
                y_test = self.get_y_test(model, self.num_test_instances)
                da_test = self.get_da_test(model, self.num_test_instances, self.num_dnase_features)
                if self.show_progress:
                    finish_bar(bar)
                    bar = Bar('Loading chromosome %s' % curr_chr, fill='#', suffix='%(percent)d%%', max=self.num_test_instances)
                idx = 0

            elif curr_chr != '-1' and curr_chr != chromosome:
                finish_bar(bar)
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
                S_test[idx] = self.get_S_data(model, shape_features, self.bin_size)
                da_test[idx] = dnase_features
                if self.show_progress:
                    bar.next()
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
    parser.add_argument('--regression', '-reg', help='Use the chipseq signal strength as targets', action='store_true', required=False)
    parser.add_argument('--show_progress', '-sp', help="show progress in progress bar", action='store_true', required=False)
    args = parser.parse_args()
    model = None

    num_epochs = 1 if args.num_epochs is None else int(args.num_epochs)
    config = int(1 if args.config is None else args.config)
    bin_size = int(200 if args.bin_size is None else args.bin_size)
    bin_size -= bin_size % 2

    num_dnase_features = 1+3+bin_size/10

    if args.model == 'TFC':
        model = ConvNet('../log/', num_epochs=num_epochs, batch_size=512,
                        num_gen_expr_features=32, config=config, dropout_rate=0.25,
                        eval_size=0.2, num_shape_features=4, sequence_width=bin_size,
                        num_dnase_features=num_dnase_features)

    else:
        print "Model options: TFC (TensorFlow Convnet)"

    unbound_fraction = 1

    if args.unbound_fraction is not None:
        unbound_fraction = float(args.unbound_fraction)

    if model is not None:
        transcription_factors = args.transcription_factors.split(',')
        evaluator = Evaluator('../data/', bin_size=bin_size,
                              ambiguous_as_bound=args.ambiguous_bound, regression=args.regression,
                              show_progress=args.show_progress, num_dnase_features=num_dnase_features)
        for transcription_factor in transcription_factors:
            if args.validate:
                evaluator.run_cross_cell_benchmark(model, transcription_factor, save_train_set=True,
                                                   unbound_fraction=unbound_fraction,
                                                   arguments=str(vars(args)).replace('\'', ''))
            if args.ladder:
                evaluator.make_ladder_predictions(model, transcription_factor, unbound_fraction=unbound_fraction, leaderboard=True)

            if args.test:
                evaluator.make_ladder_predictions(model, transcription_factor, unbound_fraction=unbound_fraction, leaderboard=False)
