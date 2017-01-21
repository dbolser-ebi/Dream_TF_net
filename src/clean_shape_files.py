if __name__ == '__main__':
    for shape_param in ['Roll', 'HelT', 'ProT', 'MGW']:
        with open('../data/annotations/hg19.genome.fa.%s' % shape_param) as fin, \
                open('../data/annotations/hg19.genome.fa.%s2' % shape_param, 'w') as fout:
            first = False
            skip = False
            for line in fin:
                if 'chr' in line:
                    if len(line) > 10:
                        skip = True
                        continue
                    elif skip:
                        skip = False

                    if first:
                        first = False
                    else:
                        fout.write('\n')
                    fout.write(line)
                else:
                    if skip:
                        continue
                    else:
                        fout.write(line.replace('NA', '0000').replace('\n', ',').strip())

