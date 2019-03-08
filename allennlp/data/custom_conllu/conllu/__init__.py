from __future__ import unicode_literals

from .models import TokenList
from .parser import parse_token_and_metadata


def conllu_parse(data, fields=None):
    return [
        TokenList(*parse_token_and_metadata(sentence, fields=fields))
        for sentence in data.split("\n\n")
        if sentence
    ]

def parse_incr(in_file, fields=None):
    for sentence in _iter_sents(in_file):
        yield TokenList(*parse_token_and_metadata(sentence, fields=fields))

def parse_tree(data):
    tokenlists = parse(data)

    sentences = []
    for tokenlist in tokenlists:
        sentences.append(tokenlist.to_tree())

    return sentences

def parse_tree_incr(in_file):
    for tokenlist in parse_incr(in_file):
        yield tokenlist.to_tree()

def _iter_sents(in_file):
    buf = []
    for line in in_file:
        if line == "\n":
            yield "".join(buf)[:-1]
            buf = []
        else:
            buf.append(line)
    if buf:
        yield "".join(buf)
