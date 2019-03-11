import logging, re
from typing import Dict, List, Iterable
from collections import defaultdict
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from itertools import repeat


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def clean_arabic(word: str):
    """
    This is to handle the strange Arabic text in Ontonotes, which has many
    more diacritics than most Arabic text found "in the wild".
    """
    diacritics = [chr(1614),chr(1615),chr(1616),chr(1617),chr(1618),chr(1761),chr(1619),chr(1648),chr(1649),chr(1611),chr(1612),chr(1613)]
    for dia in diacritics:
        word = re.sub(dia,'',word)
    return word

def _normalize_word(word: str, lang: str):
    if lang == 'ara':
        word = word.split('#')[0]
        word = clean_arabic(word)
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

@DatasetReader.register("srl")
class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)
        lang_to_instances = defaultdict(list)
        for sentence, lang in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(_normalize_word(t, lang)) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags, lang)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, tags, lang)


    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                path_components = conll_file.split('/')
                lang = None
                if 'arabic' in path_components or 'ara' in path_components:
                    lang = 'ara'
                elif 'english' in path_components or 'eng' in path_components:
                    lang = 'eng'
                elif 'chinese' in path_components or 'cmn' in path_components:
                    lang = 'cmn'
                if lang is None:
                    raise NameError("Lang name could not be automatically detected from file path. "
                                    "Check universal_dependencies.py to add language names or "
                                    "change how they are detected.")
                yield from zip(ontonotes_reader.sentence_iterator(conll_file), repeat(lang))


    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None,
                         lang: str = 'eng') -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
                                            "verb": verb, 'lang': lang})
        return Instance(fields)
