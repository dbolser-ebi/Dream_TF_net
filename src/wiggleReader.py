import argparse
from subprocess import call, Popen, PIPE
import re
from os import remove


# Low mem overhead splitting
def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[ \t\f\v0-9A-Za-z,.=']+", string))


def get_wiggle_output(fpath):
    # Call wiggletools
    process = Popen(["wiggletools", "seek", "chr10", "1000", "2000", fpath],
                    stdout=PIPE)
    (output, _) = process.communicate()
    return output


def wiggleToBedGraph(fpath):
    # Call bigWigToBedGraph
    process = call(['bigWigToBedGraph',
                    fpath, fpath+'.bedGraph'])
    return fpath+'.bedGraph'


def get_wiggle_statistics(fpath):

    '''
    Prints statistics on the wiggle file
    :param fpath: Path to wiggle file
    '''
    chromosomes = set()
    chr_range = {}  # num -> {start, end}
    total_read = 0
    total_missing = 0
    continuous_regions = 0

    bgPath = wiggleToBedGraph(fpath)
    end_pos = -1

    with open(bgPath) as fBedGraph:
        for line in fBedGraph:
            tokens = line.strip().split()
            chromosome = tokens[0]
            start_pos = int(tokens[1])
            total_missing += max(start_pos-end_pos-1, 0)
            if start_pos > end_pos:
                continuous_regions += 1
            end_pos = int(tokens[2])
            total_read += end_pos-start_pos+1

            chromosomes.add(chromosome)
            if chromosome not in chr_range:
                chr_range[chromosome] = (start_pos, end_pos)

            else:
                (bestStart, bestStop) = chr_range[chromosome]
                chr_range[chromosome] = (min(bestStart, start_pos), max(bestStop, end_pos))

    remove(bgPath)

    return chromosomes, chr_range, total_read, total_missing, continuous_regions


def get_signal_from_wiggle(fpath):
    return

def get_peaks_from_bed(path_to_bed, path_to_wiggle):
    sorted_path = path_to_bed+'.sorted'
    output_path = 'out.txt'

    with open(path_to_bed) as bedIn, open(sorted_path, 'w') as bedOut:
        # Sort bed file
        call(["sort",
             "-k1,1",
              "-k2, 2n",
              path_to_bed
              ]
             ,
             stdout=bedOut
             )

    call(["bwtool",
                     "extract",
                     "bed",
                     sorted_path,
                     path_to_wiggle,
                     output_path
                     ]
         )

    with open(output_path) as fPeaks:
        for line in fPeaks:
            print line.strip()

    # Cleanup files
    remove(output_path)
    remove(sorted_path)


def main():
    a = get_wiggle_output('../data/chipseq_fold_change_signal/ChIPseq.HepG2.YY1.fc.signal.train.bw')
    print a

if __name__ == '__main__':
    main()
