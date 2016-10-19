from multiconv import *
from kerasmulticonv import *
from datagen import *
from performance_metrics import *
import argparse


class Evaluator:
    def __init__(self, epochs, celltypes, batch_size, num_chunks, model_name, verbose, id):

        self.datagen = DataGenerator()
        if model_name == 'TFC':
            self.model = MultiConvNet('../log/', batch_size=512 if batch_size is None else batch_size, num_epochs=1 if epochs is None else epochs,
                                      sequence_width=200, num_outputs=self.datagen.num_trans_fs,
                                 eval_size=.2, early_stopping=10, num_dnase_features=63, dropout_rate=.25,
                                 config=1, verbose=True, segment='train', learning_rate=0.001,
                                      name='multiconvnet_' + str(epochs) + str(celltypes) + str(batch_size), id=id,
                                      num_chunks=num_chunks)
        elif model_name == 'KC':
            self.model = KMultiConvNet(num_epochs=epochs, num_chunks=num_chunks, verbose=verbose, batch_size=batch_size)
        self.celltypes = celltypes

    def print_results_tf(self, trans_f, y_test, y_pred):
        trans_f_idx = self.datagen.get_trans_f_lookup()[trans_f]
        y_pred = y_pred[:, trans_f_idx]
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
        held_out_celltypes = ['MCF-7', 'SK-N-SH', 'PC-3', 'liver', 'induced_pluripotent_stem_cell']
        test_celltypes = ['MCF-7', 'SK-N-SH']
        # Training
        celltypes = self.datagen.get_celltypes()

        if self.celltypes.strip().upper() == 'One'.upper():
            print "running on one celltype"
            celltypes = ['HepG2']

        for celltype in held_out_celltypes:
            try:
                celltypes.remove(celltype)
            except:
                continue

        self.model.fit(celltypes)

        # Validation
        for celltype in test_celltypes:
            print "Running benchmark for celltype", celltype
            y_test = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
            y_pred = self.model.predict(celltype, True)
            for trans_f in self.datagen.get_trans_fs():
                if celltype not in self.datagen.get_celltypes_for_trans_f(trans_f):
                    continue
                self.print_results_tf(trans_f, y_test[:2702470], y_pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', '-ne', help='Number of epochs', required=False, type=int)
    parser.add_argument('--celltypes', '-ct', help='All/One', required=False, default='One')
    parser.add_argument('--num_chunks', '-nc', help='number of chunks to train on', type=int, default=10,
                        required=False)
    parser.add_argument('--batch_size', '-batch', help='Batch size', required=False, type=int)
    parser.add_argument('--model', '-m', help='Model TFC/KC', required=False, default='TFC')
    parser.add_argument('--verbose', help='verbose optimizer', action='store_true', required=False, default=False)
    args = parser.parse_args()
    evaluator = Evaluator(args.num_epochs, args.celltypes, args.batch_size, args.num_chunks, args.model, args.verbose,
                          re.sub('[^0-9a-zA-Z]+', "", str(vars(args))))
    evaluator.run_benchmark()
