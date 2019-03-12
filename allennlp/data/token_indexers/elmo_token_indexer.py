from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

import IPython as ipy


class PreloadedElmoTokenMapper:
    """
    Maps individual tokens to token ids, compatible with a pretrained ELMo model.
    """

    def __init__(self, word_map_path):
        self.word_map = {}
        with open(word_map_path,'r') as f:
            for line in f:
                word,lang,idx = line.split()
                self.word_map[word] = int(idx)
        self.bos_token = '<S>'
        self.eos_token = '</S>'
        self.bos_id = self.convert_word_to_id(self.bos_token)
        self.eos_id = self.convert_word_to_id(self.eos_token)

    def convert_word_to_id(self, word: str) -> List[int]:
        if word in self.word_map:
            token_id = self.word_map[word]
        else:
            token_id = self.word_map["<UNK>"]
        # +1 one for masking
        return token_id + 1


@TokenIndexer.register("elmo_token_id")
class ELMoTokenIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens as single integers, drawn from a word-int mapping
    produced by the ELMo training code. This allows compatibility with saved ELMo word embeddings.
    Adapted from SingleIdTokenIndexer.

    Parameters
    ----------
    word_map : ``str``
        The path to the saved word-index map.
    namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 word_map: str,
                 namespace: str = 'tokens',
                 lowercase_tokens: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self.word_mapper = PreloadedElmoTokenMapper(word_map)

        # Leave start/end tokens off; we'll add the embeddings in elmo.py
        if start_tokens is not None or end_tokens is not None:
            print("Got some start and end tokens? from somewhere...?")
            ipy.embed()

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, 'text_id', None) is None:
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            if getattr(token, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead.
                indices.append(token.text_id)
            else:
                text = token.text
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(self.word_mapper.convert_word_to_id(text))
        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}
