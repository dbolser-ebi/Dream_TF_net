import getpass
from Bio.motifs.jaspar.db import JASPAR5
from Bio.Seq import Seq
from pyfasta import Fasta


def get_motifs_for_tf(transcription_factor, passwd):
    JASPAR_DB_HOST = 'localhost'
    JASPAR_DB_NAME = 'JASPAR'
    JASPAR_DB_USER = 'root'
    JASPAR_DB_PASS = passwd
    jdb = JASPAR5(
                    host=JASPAR_DB_HOST,
                    name=JASPAR_DB_NAME,
                    user=JASPAR_DB_USER,
                    password=JASPAR_DB_PASS
         )
    motifs = jdb.fetch_motifs_by_name(transcription_factor)
    return motifs





