import sys
from collections import defaultdict
import glob
import IPython as ipy


def build_vocab(filepatterns):
    word_counts = defaultdict(int)
    filenames = []
    for filepattern in filepatterns:
        filenames += glob.glob(filepattern)
    for filename in filenames:
        lang = filename.split('/')[-2]
        sys.stderr.write(filename+", lang is {}\n".format(lang))
        lines = open(filename).readlines()
        for line in lines:
            for w in line.split():
                w = lang + ':' + w
                word_counts[w] += 1
    return word_counts

def print_vocab(counts, print_count=False):
    unkc = 0
    for w in counts:
        if counts[w] < 2:
            unkc += counts[w]
    print("<S>")
    print("</S>")
    print("<UNK>")
    for w in sorted(counts, key=counts.get, reverse=True):
        #if counts[w] < 2:
        #    continue
        print(w)
    sys.stdout.flush()

if __name__=="__main__":
    counts = build_vocab(sys.argv[1:])
    print_vocab(counts)


        
