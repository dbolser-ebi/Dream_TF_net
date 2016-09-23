from multiconv import *
from datagen import *
from performance_metrics import *


class Evaluator:
    def __init__(self):
        self.datagen = DataGenerator()

    def print_results_tf(self, trans_f, y_test, y_pred):
        trans_f_idx = self.datagen.get_trans_f_lookup()[trans_f]
        print "Results for transcription factor", trans_f
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
        print 'AU ROC', auroc(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten())
        print 'AU PRC', auprc(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten())
        print 'RECALL AT FDR 0.9', recall_at_fdr(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten(), 0.90)
        print 'RECALL AT FDR 0.5', recall_at_fdr(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten(), 0.50)
        print 'RECALL AT FDR 0.25', recall_at_fdr(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten(), 0.25)
        print 'RECALL AT FDR 0.1', recall_at_fdr(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten(), 0.10)
        print 'RECALL AT FDR 0.05', recall_at_fdr(y_test[trans_f_idx].flatten(), y_pred[trans_f_idx].flatten(), 0.05)

    def make_predictions(self):
        return

    def run_benchmark(self):
        datagen = DataGenerator()
        held_out_celltypes = ['MCF-7', 'SK-N-SH', 'PC-3', 'liver', 'induced_pluripotent_stem_cell']
        test_celltypes = ['MCF-7', 'SK-N-SH']
        # Training
        model = MultiConvNet('../log', batch_size=512, num_epochs=1, sequence_width=600, num_outputs=datagen.num_trans_fs,
                             eval_size=.2, early_stopping=10, num_dnase_features=13, dropout_rate=.25,
                             config=7, verbose=True, name='multiconvnet', segment='train')
        celltypes = datagen.get_celltypes()
        for celltype in held_out_celltypes:
            celltypes.remove(celltype)
        model.fit(celltypes)

        # Validation
        for celltype in test_celltypes:
            y_test = np.load(os.path.join(self.datagen.save_dir, 'y_%s.npy' % celltype))
            for trans_f in ['CTCF']:
                y_pred = model.predict(celltype)[:, self.datagen.get_trans_f_lookup()[trans_f]]
                self.print_results_tf(trans_f, y_test, y_pred)

    def compute_bound(self):
        datagen = DataGenerator()
        bound_positions = datagen.get_bound_positions()
        print len(bound_positions)

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.compute_bound()
    #evaluator.run_benchmark()
