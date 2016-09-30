import argparse
import os
import synapseclient
import getpass


def submit(folder, id, password):

    syn = synapseclient.Synapse()
    syn.login(email=id, password=password)

    evaluation = syn.getEvaluation(7071644)
    project = syn.get(7118431)

    submission_filenames = [fname for fname in os.listdir(folder)]

    for filename in submission_filenames:
        if filename.startswith('F'):
            filename = os.path.join(folder, filename)
            f_handler = synapseclient.File(filename, parent=project)
            entity = syn.store(f_handler)
            syn.submit(evaluation, entity, name='test', team='Nabla')
            break

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-u',
                           help='synapse username', required=True)

    args = argparser.parse_args()
    passwd = getpass.getpass()
    submit('../results/', args.u, passwd)