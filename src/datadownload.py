################################################################################
## Download competition data for the
## ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge
################################################################################

import synapseclient
import argparse
import multiprocessing
import threading
from dataclean import clean_folder, rename_folders


class DownloadThread(threading.Thread):

    def __init__(self, folder_id, syn, dest_dir):
        super(DownloadThread, self).__init__()
        self.folder_id = folder_id
        self.syn = syn
        self.dest_dir = dest_dir

    def run(self):
        ## get folder
        folder = self.syn.get(self.folder_id, downloadLocation=self.dest_dir+'/'+self.folder_id,
                              ifcollission='overwrite.local')
        print 'Downloading contents of %s folder (%s)\n' % (folder.name, folder.id,)

        ## query for child entities
        query_results = self.syn.query('select id,name from file where parentId=="%s"' % self.folder_id)

        ## download all genotype data files
        for entity in query_results['results']:
            print '\tDownloading file: ', entity['file.name']
            self.syn.get(entity['file.id'], downloadLocation=self.dest_dir+'/'+self.folder_id)


def main():
    multiprocessing.freeze_support()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-u',
                           help='synapse username', required=True)
    argparser.add_argument('-p',
                           help='synapse password', required=True)
    argparser.add_argument('-l',
                           help='download location', required=True)
    args = argparser.parse_args()

    syn = synapseclient.Synapse()

    syn.login(email=args.u, password=args.p)

    ## You may wish to copy these files to a specific destination directory. If so,
    ## set the path to that directory here or pass it as an argument to the script.
    dest_dir = args.l

    folder_ids = [
        'syn6181334', # ChIPseq fold_change_signal +- 200GB
        'syn6181335', # ChIPseq labels
        'syn6181337', # ChIPseq peaks conservative
        'syn6181338', # ChIPseq peaks relaxed
        #'syn6176232', # DNASE bams +- 2TB
        'syn6176233', # DNASE fold_coverage_wiggles
        'syn6176235', # DNASE peaks conservative
        'syn6176236', # DNASE peaks relaxed
        'syn6176231', # RNAseq
        'syn6184307' # annotations
                 ]

    threads = []
    for folder_id in folder_ids:
        threads.append(DownloadThread(folder_id, syn, dest_dir))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print 'Download complete!'

    print 'Cleaning...'
    clean_folder(dest_dir)

    print 'Renaming...'
    rename_folders(dest_dir)

if __name__ == '__main__':
    main()
