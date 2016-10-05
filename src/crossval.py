from performance_metrics import *
from convnet import *
import sqlite3
from datagen import *
from ensemblr import *
from bisect import bisect_left
import re
from scipy.stats import rankdata


class Evaluator:

    def __init__(self, datapath, bin_size=200, num_dnase_features=4,
                 unbound_fraction=1.0, dnase_bin_size=200, chipseq_bin_size=200, debug=False, num_train_celltypes=10):
        self.num_dnase_features = num_dnase_features
        self.num_train_instances = 51676736
        self.num_tf = 32
        self.num_celltypes = 14
        self.num_train_celltypes = 11
        self.datapath = datapath
        self.bin_size = bin_size
        self.unbound_fraction = unbound_fraction
        self.chipseq_bin_size = chipseq_bin_size
        self.dnase_bin_size = dnase_bin_size
        self.num_test_instances = 0
        self.debug = debug
        self.num_train_celltypes = num_train_celltypes
        self.datagen = DataGenerator()

    def print_results(self, y_test, y_pred):
        y_pred = rankdata(y_pred).astype(np.float32)/len(y_pred)
        print 'AU ROC', auroc(y_test.flatten(), y_pred.flatten())
        print 'AU PRC', auprc(y_test.flatten(), y_pred.flatten())
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)

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


    def make_ladder_predictions(self, model, transcription_factor, unbound_fraction=1.0, leaderboard=True):
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
            'ZNF143': ['K562']
        }
        tf_final = {
            'ATF2': ['HepG2'],
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

        celltypes = self.datagen.get_celltypes_for_tf(transcription_factor)

        model.set_transcription_factor(transcription_factor)

        model.fit_combined(celltypes)

        if leaderboard:
            print "Creating predictions for leaderboard"
        else:
            print "Creating predictions for final round"

        tf_mapper = tf_leaderboard if leaderboard else tf_final

        part = 'ladder' if leaderboard else 'test'

        if leaderboard:
            num_test_indices = 8843011
        else:
            num_test_indices = 60519747

        segment = 'ladder' if leaderboard else 'test'

        if transcription_factor not in tf_mapper:
            return

        for test_celltype in tf_mapper[transcription_factor]:

            dnase_features = np.load('../data/preprocess/DNASE_FEATURES_NORM/%s_%s_%d.gz.npy'
                                         % (test_celltype, part, self.dnase_bin_size))
            y_pred = []
            stride = 1000000
            for start in range(0, num_test_indices, stride):
                ids = range(start, min(start + stride, num_test_indices))
                X = self.datagen.get_sequece_from_ids(ids, segment)
                y_pred.extend(list(model.predict_combined(X, dnase_features)))

            fin = gzip.open(os.path.join(self.datapath, 'annotations/%s_regions.blacklistfiltered.bed.gz' % part))

            if leaderboard:
                f_out_name = '../results/' + 'L.' + transcription_factor + '.' + test_celltype + '.tab.gz'
            else:
                f_out_name = '../results/' + 'F.' + transcription_factor + '.' + test_celltype + '.tab.gz'

            with gzip.open(f_out_name, 'w') as fout:
                for idx, line in enumerate(fin):
                    print>> fout, str(line.strip()) + '\t' + str(y_pred[idx])

            fin.close()

    def run_cross_cell_benchmark(self, model, transcription_factor, arguments=""):

        print "Running cross celltype benchmark for transcription factor %s" % transcription_factor


        #--------------- TRAIN
        celltypes = self.datagen.get_celltypes_for_tf(transcription_factor)
        #random.shuffle(celltypes)

        model.set_transcription_factor(transcription_factor)

        celltypes_train = celltypes[1:1+self.num_train_celltypes]
        celltype_test = celltypes[0]

        model.fit_combined(celltypes_train)

        print 'TRAINING COMPLETED'

        # --------------- VALIDATION
        print
        print "RUNNING TESTS"

        print "Validating on celltype", celltype_test

        ids = range(2702470) #chr10
        X = self.datagen.get_sequece_from_ids(ids, 'train', self.bin_size)
        dnase_features = self.datagen.get_dnase_features_from_ids(ids, 'train', celltype_test, dnase_bin_size=self.bin_size)

        trans_f_lookup = self.datagen.get_trans_f_lookup()
        y_test = np.load('../data/preprocess/features/y_%s.npy' % celltype_test)[ids, trans_f_lookup[transcription_factor]]
        y_pred = model.predict_combined(X, dnase_features)
        if self.debug:
            with open('../data/debug/predictions_' +
                              celltype_test + transcription_factor + model.__class__.__name__ + 'chr10' + '.bedgraph', 'w') \
                    as fout, \
                    gzip.open('../data/annotations/train_regions.blacklistfiltered.bed.gz') as fin:

                for idx, line in enumerate(fin):
                    tokens = line.split()
                    if idx >= len(ids):
                        break
                    start = tokens[1]
                    end = tokens[2]
                    print>> fout, 'chr10', start, end, y_pred[idx]

        self.print_results(y_test, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription_factors', '-tfs', help='Comma separated list of transcription factors', required=True)
    parser.add_argument('--model', '-m', help='Choose model [TFC/RENS]', required=True)
    parser.add_argument('--regression', '-reg', help='Use the chipseq signal strength as targets', action='store_true',
                        required=False)
    parser.add_argument('--validate', '-v', action='store_true', help='run cross TF validation benchmark', required=False)
    parser.add_argument('--ladder', '-l', action='store_true', help='predict TF ladderboard', required=False)
    parser.add_argument('--test', '-t', action='store_true', help='predict TF final round', required=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=512, required=False)
    parser.add_argument('--num_chunks', '-nc', help='number of chunks to train on', type=int, default=10, required=False)
    parser.add_argument('--config', '-c', help='configuration of model', default=7, type=int, required=False)
    parser.add_argument('--num_epochs', '-ne', help='number of epochs', type=int, default=1, required=False)
    parser.add_argument('--bin_size', '-bs', help='Sequence bin size (must be an even number >= 200)', type=int, default=200, required=False)
    parser.add_argument('--dnase_bin_size', '-dbs', help='DNASE bin size', type=int, default=200, required=False)
    parser.add_argument('--chipseq_bin_size', '-cbs', help='CHIPSEQ bin size', type=int, default=200, required=False)
    parser.add_argument('--debug', '-dbg', help='Debug the model', action='store_true', required=False)
    parser.add_argument('--num_train_celltypes', '-ntc', help='Number of celltypes to use for training', type=int, default=10, required=False)

    args = parser.parse_args()
    model = None

    if args.model == 'TFC':
        model = ConvNet('../log/', num_epochs=args.num_epochs, batch_size=args.batch_size,
                        num_gen_expr_features=32, config=args.config, dropout_rate=0.25,
                        eval_size=0.2, num_shape_features=4, sequence_width=args.bin_size, regression=args.regression,
                        num_chunks=args.num_chunks, id=re.sub('[^0-9a-zA-Z]+', "", str(vars(args))))

    else:
        print "Model options: TFC (TensorFlow Convnet) / RENS (Random forest classifier ensemble)"

    transcription_factors = args.transcription_factors.split(',')
    evaluator = Evaluator('../data/', bin_size=args.bin_size, dnase_bin_size=args.bin_size,
                          chipseq_bin_size=args.chipseq_bin_size, debug=args.debug, num_train_celltypes=args.num_train_celltypes)
    for transcription_factor in transcription_factors:
        if args.validate:
            evaluator.run_cross_cell_benchmark(model, transcription_factor,
                                               arguments=str(vars(args)).replace('\'', ''))
        if args.ladder:
            evaluator.make_ladder_predictions(model, transcription_factor, leaderboard=True)

        if args.test:
            evaluator.make_ladder_predictions(model, transcription_factor, leaderboard=False)
