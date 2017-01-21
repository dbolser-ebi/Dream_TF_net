from multiconv import *
from kerasmulticonv import *
from datagen import *
from performance_metrics import *
import argparse


class Evaluator:
    def __init__(self, num_epochs, batch_size, model_name, verbose, id, sequence_bin_size=200, dnase_bin_size=200,
                 num_train_celltypes=14, randomize_celltypes=False, config=1):

        self.num_train_celltypes = num_train_celltypes

        self.datagen = DataGenerator()
        '''
        if model_name == 'TFC':
            self.model = MultiConvNet('../log/', batch_size=512 if batch_size is None else batch_size, num_epochs=1 if epochs is None else epochs,
                                      sequence_width=200, num_outputs=self.datagen.num_trans_fs,
                                 eval_size=.2, early_stopping=10, num_dnase_features=63, dropout_rate=.25,
                                 config=1, verbose=True, segment='train', learning_rate=0.001,
                                      name='multiconvnet_' + str(epochs) + str(celltypes) + str(batch_size), id=id)
        '''
        if model_name == 'KC':
            self.model = KMultiConvNet(num_epochs=num_epochs, verbose=verbose,
                                       batch_size=batch_size, sequence_bin_size=sequence_bin_size,
                                       dnase_bin_size=dnase_bin_size, randomize_celltypes=randomize_celltypes, config=config)

    def print_results_tf(self, trans_f, y_test, y_pred):
        trans_f_idx = self.datagen.get_trans_f_lookup()[trans_f]
        y_pred = y_pred[trans_f_idx][:, 2]
        y_test = y_test[:, trans_f_idx]
        print "Results for transcription factor", trans_f
        print 'AU ROC', auroc(y_test.flatten(), y_pred.flatten())
        print 'AU PRC', auprc(y_test.flatten(), y_pred.flatten())
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)

    def make_predictions(self):

        for leaderboard in [True, False]:
            segment = 'ladder' if leaderboard else 'test'
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
            tf_lookup = self.datagen.get_trans_f_lookup()

            inv_mapping = {}
            for transcription_factor in tf_mapper.keys():
                for celltype in tf_mapper[transcription_factor]:
                    if celltype not in inv_mapping:
                        inv_mapping[celltype] = []
                    inv_mapping[celltype].append(transcription_factor)

            for test_celltype in inv_mapping.keys():
                y_pred = self.model.predict(test_celltype)
                for transcription_factor in inv_mapping[test_celltype]:
                    if leaderboard:
                        f_out_name = '../results/' + 'L.' + transcription_factor + '.' + test_celltype + '.tab.gz'
                    else:
                        f_out_name = '../results/' + 'F.' + transcription_factor + '.' + test_celltype + '.tab.gz'

                    fin = gzip.open(os.path.join(self.datagen.datapath),
                                    'annotations/%s_regions.blacklistfiltered.bed.gz'
                                    % ('ladder' if leaderboard else 'test'))

                    with gzip.open(f_out_name, 'w') as fout:
                        for idx, line in enumerate(fin):
                            print>> fout, str(line.strip()) + '\t' + str(y_pred[idx, tf_lookup[transcription_factor]])
                    fin.close()

    def run_benchmark(self):
        held_out_celltypes = ['MCF-7', 'PC-3', 'liver', 'induced_pluripotent_stem_cell']
        test_celltypes = ['MCF-7']
        # Training
        train_celltypes = self.datagen.get_celltypes()

        if 'SK-N-SH' in train_celltypes:
            train_celltypes.remove('SK-N-SH')

        for celltype in held_out_celltypes:
            try:
                train_celltypes.remove(celltype)
            except:
                continue

        train_celltypes = train_celltypes[:self.num_train_celltypes]

        self.model.fit(train_celltypes)

        # Validation
        for celltype in test_celltypes:
            print "Running benchmark for celltype", celltype
            y_test = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
            y_pred = self.model.predict(celltype, 'train', True)
            for trans_f in self.datagen.get_trans_fs():
                if celltype not in self.datagen.get_celltypes_for_trans_f(trans_f):
                    continue
                self.print_results_tf(trans_f, y_test[:2702470], y_pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', '-v', action='store_true', help='run cross TF validation benchmark',
                        required=False)
    parser.add_argument('--ladder', '-l', action='store_true', help='predict TF ladderboard', required=False)
    parser.add_argument('--test', '-t', action='store_true', help='predict TF final round', required=False)
    parser.add_argument('--batch_size', '-batch', help='Batch size', required=False, type=int, default=512)
    parser.add_argument('--model', '-m', help='Model KC', required=False, default='KC')
    parser.add_argument('--verbose', help='verbose optimizer', action='store_true', required=False, default=False)
    parser.add_argument('--num_train_celltypes', '-ntc', help='Number of celltypes to use for training', type=int, default=10, required=False)
    parser.add_argument('--randomize_celltypes', '-rc', help='Randomize celltypes over batches', action='store_true', required=False)
    parser.add_argument('--num_epochs', '-ne', help='number of epochs', type=int, default=1, required=False)
    parser.add_argument('--sequence_bin_size', '-sbs', help='Sequence bin size (must be an even number >= 200)',
                        type=int, default=200, required=False)
    parser.add_argument('--dnase_bin_size', '-dbs', help='DNASE bin size', type=int, default=200, required=False)
    parser.add_argument('--chipseq_bin_size', '-cbs', help='CHIPSEQ bin size', type=int, default=200, required=False)
    parser.add_argument('--config', '-c', help='configuration of model', default=7, type=int, required=False)

    args = parser.parse_args()
    evaluator = Evaluator(num_epochs=args.num_epochs,
                          batch_size=args.batch_size,
                          model_name=args.model,
                          verbose=args.verbose,
                          id=re.sub('[^0-9a-zA-Z]+', "", str(vars(args))),
                          sequence_bin_size=args.sequence_bin_size,
                          dnase_bin_size=args.dnase_bin_size,
                          num_train_celltypes=args.num_train_celltypes,
                          randomize_celltypes=args.randomize_celltypes,
                          config=args.config)
    if args.validate:
        evaluator.run_benchmark()
