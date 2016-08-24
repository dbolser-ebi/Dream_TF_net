'''
Clean the specified data directory after using datadownload script
'''
import argparse
import os
import re
import shutil


def make_dnase_peak_naming_consistent(fpath):
    if os.path.exists(fpath+'DNASE.IMR90.conservative.narrowPeak.gz'):
        print 'Renaming', 'DNASE.IMR90.conservative.narrowPeak.gz'
        shutil.move(fpath+'DNASE.IMR90.conservative.narrowPeak.gz',
                    fpath+'DNASE.IMR-90.conservative.narrowPeak.gz')


def make_dnase_fold_naming_consistent(fpath):
    if os.path.exists(os.path.join(fpath, 'DNASE.IMR90.fc.signal.bigwig')):
        print 'Renaming', 'DNASE.IMR90.fc.signal.bigwig'
        shutil.move(os.path.join(fpath, 'DNASE.IMR90.fc.signal.bigwig'),
                    os.path.join(fpath, 'DNASE.IMR-90.fc.signal.bigwig'))


def make_rnaseq_naming_consistent(fpath):
    if os.path.exists(fpath+'gene_expression.IMR90.biorep1.tsv'):
        print 'Renaming', 'gene_expression.IMR90.biorep1.tsv'
        shutil.move(fpath+'gene_expression.IMR90.biorep1.tsv',
                    fpath+'gene_expression.IMR-90.biorep1.tsv')
    if os.path.exists(fpath+'gene_expression.IMR90.biorep2.tsv'):
        print 'Renaming', 'gene_expression.IMR90.biorep2.tsv'
        shutil.move(fpath+'gene_expression.IMR90.biorep2.tsv',
                    fpath+'gene_expression.IMR-90.biorep2.tsv')


def make_naming_consistent(fpath):
    make_dnase_peak_naming_consistent(os.path.join(fpath,'dnase_peaks_conservative/'))
    make_dnase_fold_naming_consistent(os.path.join(fpath,'dnase_fold_coverage/'))
    make_rnaseq_naming_consistent(os.path.join(fpath,'rnaseq/'))


def clean_duplicate(num, path):
    pattern = r'.+\(%s\)' % str(num)
    match = re.search(pattern, path)
    if match:
        print "MATCHED", path
        new_path = path.replace('(%s)' % str(num), '')
        shutil.move(path, new_path)
        print "    MOVED TO", new_path


def clean_folder(fpath):
    for directory in os.listdir(fpath):
        dirpath = os.path.join(fpath, directory)
        if os.path.isdir(dirpath):
            for i in range(10):
                for file in os.listdir(dirpath):
                    filepath = os.path.join(dirpath, file)
                    clean_duplicate(i, filepath)


def rename_folders(fpath):
    name_map = {
        'syn6181334': 'chipseq_fold_change_signal',
        'syn6181335': 'chipseq_labels',
        'syn6181337': 'chipseq_conservative_peaks',
        'syn6181338': 'chipseq_relaxed_peaks',
        'syn6176232': 'dnase_bams',
        'syn6176233': 'dnase_fold_coverage',
        'syn6176235': 'dnase_peaks_conservative',
        'syn6176236': 'dnase_peaks_relaxed',
        'syn6176231': 'rnaseq',
        'syn6184307': 'annotations'
    }

    for directory in os.listdir(fpath):
        dirpath = os.path.join(fpath, directory)
        if os.path.isdir(dirpath) and directory in name_map.keys():
            print 'Moving', directory, 'to', name_map[directory]
            shutil.move(dirpath, os.path.join(fpath, name_map[directory]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to data directory (i.e. ../data/)', required=True)
    args = parser.parse_args()
    clean_folder(args.path)
    rename_folders(args.path)
    make_naming_consistent(args.path)

if __name__ == '__main__':
    main()
