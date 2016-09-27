from multiconv import *
from datagen import *
from performance_metrics import *
import argparse


class Evaluator:
    def __init__(self, mode):
        self.datagen = DataGenerator()
        self.model = MultiConvNet('../log', batch_size=512, num_epochs=1, sequence_width=600, num_outputs=self.datagen.num_trans_fs,
                             eval_size=.2, early_stopping=10, num_dnase_features=13, dropout_rate=.25,
                             config=7, verbose=True, name='multiconvnet_'+mode, segment='train', learning_rate=0.001)
        self.mode = mode

    def print_results_tf(self, trans_f, y_test, y_pred):
        trans_f_idx = self.datagen.get_trans_f_lookup()[trans_f]
        y_pred = y_pred[:, trans_f_idx]
        y_test = y_test[:, trans_f_idx]
        print "Results for transcription factor", trans_f
        print 'AU ROC', auroc(y_test.flatten(), y_pred.flatten())
        print 'AU PRC', auprc(y_test.flatten(), y_pred.flatten())
        print 'RECALL AT FDR 0.9', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.90)
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.50)
        print 'RECALL AT FDR 0.25', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.25)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.10)
        print 'RECALL AT FDR 0.05', recall_at_fdr(y_test.flatten(), y_pred.flatten(), 0.05)

    def make_predictions(self, leaderboard=True):
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
        if self.mode == 'FA':
            celltypes = self.datagen.get_celltypes()
        elif self.mode == 'FTF':
            celltypes = self.datagen.get_celltypes_for_trans_f('CTCF')

        for celltype in held_out_celltypes:
            try:
                celltypes.remove(celltype)
            except:
                continue

        celltypes = celltypes[:4]

        if self.mode == 'FA':
            self.model.fit_all(celltypes)
        elif self.mode == 'FTF':
            self.model.fit_tf(celltypes)

        # Validation
        for celltype in test_celltypes:
            print "Running benchmark for celltype", celltype
            y_test = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
            y_pred = self.model.predict(celltype)
            for trans_f in self.datagen.get_trans_fs():
                if celltype not in self.datagen.get_celltypes_for_trans_f(trans_f):
                    continue
                self.print_results_tf(trans_f, y_test[:2702470], y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='FA/FTF')
    args = parser.parse_args()
    evaluator = Evaluator(args.mode)
    evaluator.run_benchmark()
