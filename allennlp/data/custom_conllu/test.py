import IPython as ipy
from conllu import parse
with open('test_parser.conllu') as fin:
    data = fin.read()
sents = parse(data)
ipy.embed()
