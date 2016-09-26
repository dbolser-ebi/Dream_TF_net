from multiconv import *
from datagen import *
from performance_metrics import *
import argparse


class Evaluator:
    def __init__(self):
        self.datagen = DataGenerator()
        self.model = MultiConvNet('../log', batch_size=512, num_epochs=5, sequence_width=600, num_outputs=self.datagen.num_trans_fs,
                             eval_size=.2, early_stopping=10, num_dnase_features=13, dropout_rate=.25,
                             config=7, verbose=True, name='multiconvnet', segment='train', learning_rate=0.001)

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

    def make_predictions(self):
        #TODO
        held_out_celltypes = ['PC-3', 'liver', 'induced_pluripotent_stem_cell']
        celltypes = self.datagen.get_celltypes()
        for celltype in held_out_celltypes:
            try:
                celltypes.remove(celltype)
            except:
                continue

    def run_benchmark(self, mode):
        held_out_celltypes = ['MCF-7', 'SK-N-SH', 'PC-3', 'liver', 'induced_pluripotent_stem_cell']
        test_celltypes = ['MCF-7', 'SK-N-SH']
        # Training
        if mode == 'FA':
            celltypes = self.datagen.get_celltypes()
        elif mode == 'FTF':
            celltypes = self.datagen.get_celltypes_for_trans_f('CTCF')[:2]

        for celltype in held_out_celltypes:
            try:
                celltypes.remove(celltype)
            except:
                continue
        if mode == 'FA':
            self.model.fit_all(celltypes)
        elif mode == 'FTF':
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
    evaluator = Evaluator()
    evaluator.run_benchmark(args.mode)
