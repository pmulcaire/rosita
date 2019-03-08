# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random
from collections import defaultdict
import os
import sys

import numpy as np

from typing import List

import IPython as ipy

random.seed(3243)

class Vocabulary(object):
    """
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    """
    def __init__(self, filename, validate_file=False):
        """
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        """
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos= idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self._eos] + word_ids + [self._bos], dtype=np.int32)
        else:
            return np.array([self._bos] + word_ids + [self._eos], dtype=np.int32)



class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self._bos] = self.bos_chars
        self._word_char_ids[self._eos] = self.eos_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])



class PolyglotCharsVocabulary(UnicodeCharsVocabulary):
    """Vocabulary containing character-level and word level information
    from multiple languages.

    Has a word vocabulary that is used to look up word ids and a character
    vocabulary that is used to map word + language name to an array of
    character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, validate_file=False):
        """
        __init__ doesn't call super() (since the default Vocabulary is not 
        polyglot) so initialization code is copied and rewritten. Some things
        in initialization are incompatiable with the Vocabulary class or the
        UnicodeCharsVocabulary class, but we still want to inherit methods.
        
        filename = a flat text file with one (normalized) token per line. Each
            token should be prefixed with a language code (e.g. eng:dog). In
            addition, the file should also contain the special tokens <S>, 
            </S>, <UNK> (unprefixed, case sensitive).
        """
        self._id_to_wordl = [] #wordl = (word, lang) pair
        self._wordl_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1
        self.special_tokens = ['<S>','</S>','<UNK>']

        idx = 0
        if isinstance(filename,list):
            print("Multiple vocab files? this will not work; sampled softmax "
                  "means full vocab must be sorted by frequency")
            raise ValueError
        with open(filename) as f:
            for line in f:
                word_name = line.strip()
                if word_name == '<S>' and self._bos == -1:
                    self._bos = idx
                elif word_name == '</S>' and self._eos == -1:
                    self._eos = idx
                elif word_name == '<UNK>' and self._unk == -1:
                    self._unk = idx

                if word_name in self.special_tokens:
                    if word_name not in self._wordl_to_id:
                        self._id_to_wordl.append(word_name)
                        self._wordl_to_id[word_name] = idx
                        idx += 1
                    continue
                if word_name == '!!!MAXTERMID':
                    continue

                # handle language prefixes
                lang = word_name.split(':')[0]
                word_name = ':'.join(word_name.split(':')[1:])

                self._id_to_wordl.append((word_name, lang))
                self._wordl_to_id[(word_name, lang)] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

        self._max_word_length = max_word_length

        """
        Character representation is all polyglot, no language IDs
        """
        # count unique characters
        char_counts = defaultdict(int)
        for i, word_lang_pair in enumerate(self._id_to_wordl):
            try:
                word, lang = word_lang_pair
            except:
                # special tokens with no lang still get character processing
                word = word_lang_pair
            for ch in word:
                char_counts[ch] += 1
        self.char_ids = {ch:i for i,ch in enumerate(sorted(char_counts.keys()))}

        # assign next 5 ids to special chars
        n_chars = len(char_counts)
        self.bos_char = n_chars  # <begin sentence>
        self.eos_char = n_chars+1  # <end sentence>
        self.bow_char = n_chars+2  # <begin word>
        self.eow_char = n_chars+3  # <end word>
        self.pad_char = n_chars+4  # <padding>
        self.unk_char = n_chars+5  # <unseen chars -- needed since char reprs are based on vocab>
        self.n_chars = n_chars+6

        num_words = len(self._id_to_wordl)

        self._word_char_ids = np.zeros([num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word_lang_pair in enumerate(self._id_to_wordl):
            try:
                word, lang = word_lang_pair
            except:
                #for special tokens with no lang
                word = word_lang_pair
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self._bos] = self.bos_chars
        self._word_char_ids[self._eos] = self.eos_chars


    def save_vocab(self, savedir):
        """
        Save the character and word vocabularies to file (so that the allennlp
        ELMo code can properly associate characters with their IDs.
        """
        with open(os.path.join(savedir,"char_vocab.txt"),'w') as f:
            for w in self.char_ids:
                f.write("{}\t{}\n".format(w,self.char_ids[w]))
        with open(os.path.join(savedir,"word_vocab.txt"),'w') as f:
            for key in self._wordl_to_id:
                if key in self.special_tokens:
                    word,lang = key,'_'
                else:
                    word,lang = key
                f.write("{}\t{}\t{}\n".format(word,lang,self._wordl_to_id[key]))
        sys.stderr.write("Wrote char and word vocabs to file")               
            

    @property
    def unk(self):
        """
        Currently unchanged, but could become language specific and take 
        langname as an argument.
        """
        return self._unk

    @property
    def size(self):
        return len(self._id_to_wordl)

    def word_to_id(self, word):
        raise NotImplementedError("Need a language name too")

    def word_to_id(self, word, lang):
        if word in self.special_tokens:
            return self._wordl_to_id[word]
        if (word, lang) in self._wordl_to_id:
            return self._wordl_to_id[(word, lang)]
        return self.unk

    def id_to_word(self, cur_id):
        if cur_id in [self._bos, self._eos, self._unk]:
            return (self._id_to_wordl[cur_id], None)
        return self._id_to_wordl[cur_id]

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        code[0] = self.bow_char
        for k, ch in enumerate(word[:(self.max_word_length-2)], start=1):
            if ch in self.char_ids:
                code[k] = self.char_ids[ch]
            else:
                code[k] = self.unk_char
        code[k+1] = self.eow_char
        return code

    def decode(self, cur_ids, prefixed=False):
        """Convert a list of ids to a sentence, with space inserted."""
        sentence = []
        for cur_id in cur_ids:
            word, lang = self.id_to_word(cur_id)
            if prefixed and lang is not None:
                word = lang + ':' + word
            sentence.append(word)
        return ' '.join(sentence)

    def encode(self, sentence, lang, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word,lang) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word,lang) for cur_word in sentence]

        if reverse:
            return np.array([self._eos] + word_ids + [self._bos], dtype=np.int32)
        else:
            return np.array([self._bos] + word_ids + [self._eos], dtype=np.int32)

    def word_to_char_ids(self, word, lang):
        if (word, lang) in self._wordl_to_id:
            return self._word_char_ids[self._wordl_to_id[(word,lang)]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, lang, reverse=False, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [self.word_to_char_ids(cur_word, lang)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word, lang)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


class Batcher(object):
    """ 
    Batch sentences of tokenized text into character id matrices.
    """
    def __init__(self, lm_vocab_file: str, max_token_length: int):
        """
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        """
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length
        )
        self._max_token_length = max_token_length

    def batch_sentences(self, sentences: List[List[str]]):
        """
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        """
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object):
    """ 
    Batch sentences of tokenized text into token id matrices.
    """
    def __init__(self, lm_vocab_file: str):
        """
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        """
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences: List[List[str]]):
        """
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        """
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        return X_ids


##### for training
def _get_batch(generator, batch_size, num_steps, max_word_length):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_word_length is not None:
            char_inputs = np.zeros([batch_size, num_steps, max_word_length],
                                np.int32)
        else:
            char_inputs = None
        targets = np.zeros([batch_size, num_steps], np.int32)

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                                                                    :how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]

                cur_pos = next_pos

                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

        if no_more_data:
            # There is no more data.  Note: this will not return data
            # for the incomplete batch
            break

        X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                 'next_token_id': targets}

        yield X

class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.  Each file contains one sentence
        per line.  Each sentence is pre-tokenized and white space joined.
    """
    def __init__(self, filepaths, vocab, reverse=False, test=False, dev=False,
                 shuffle_on_load=False):
        """
        filepaths = a list that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        """
        self._vocab = vocab
        self._all_shards = [path for path in filepaths]
        self._file_paths = [path for path in filepaths]
        print('Found %d shards at %s' % (len(self._all_shards), filepaths))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                print("Used all shards")
                try:
                    if self._dev:
                        # put all shards back so we can iterate again later
                        self._all_shards = self.file_paths
                except AttributeError:
                    pass
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)

        ids = [self._vocab.encode(sentence, self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self._vocab.encode_chars(sentence, self._reverse)
                     for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        #print('Loaded %d sentences.' % len(ids))
        return list(zip(ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(self.get_sentence(), batch_size, num_steps,
                           self.max_word_length):

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab


class PolyglotLMDataset(LMDataset):
    """
    subclass to deal with language prefixes in filepaths and vocabulary
    """
    def __init__(self, filepaths, vocab, reverse=False, test=False, dev=False,
                 shuffle_on_load=False):
        self._lang_to_shards = defaultdict(list)
        self._lang_shards_to_choose = defaultdict(list)
        self._file_paths = [path for path in filepaths]
        for shard_name in filepaths:
            language_name = shard_name.split('/')[-2]
            self._lang_to_shards[language_name].append(shard_name)
        self._nlangs = len(self._lang_to_shards)
        self._langs = list(self._lang_to_shards.keys())
        self._lang_counter = 0
        super().__init__(filepaths, vocab, reverse, test, shuffle_on_load)

    def _choose_random_shard(self, lang):
        if len(self._lang_shards_to_choose[lang]) == 0:
            self._lang_shards_to_choose[lang] = list(self._lang_to_shards[lang])
            random.shuffle(self._lang_shards_to_choose[lang])
        shard_name = self._lang_shards_to_choose[lang].pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                print("Used all shards")
                try:
                    if self._dev:
                        # put all shards back so we can iterate again later
                        self._all_shards = glob.glob(self.file_paths)
                except AttributeError:
                    pass
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard from the next lang
            self._lang_counter += 1
            self._lang_counter %= self._nlangs
            lang = self._langs[self._lang_counter]
            shard_name = self._choose_random_shard(lang)
        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """
        Read one file and convert to ids.
        Args:
            shard_name: file path.
        Returns:
            list of (id, char_id) tuples.

        Currently the same as the monolingual version except that it detects the
        language of the data it's loading and includes that in the calls to vocab.encode()
        (and vocab.encode_chars(), which is polyglot, but uses lang for caching, uselessly)
        """
        print('Loading data from: %s' % shard_name, end='; ')
        language_name = shard_name.split('/')[-2]
        print('language is %s' % language_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)

        ids = [self._vocab.encode(sentence, lang=language_name, reverse=self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self._vocab.encode_chars(sentence, lang=language_name, reverse=self._reverse)
                         for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        return list(zip(ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret


class BidirectionalLMDataset(object):
    def __init__(self, filepaths, vocab, test=False, shuffle_on_load=False):
        """
        bidirectional version of LMDataset
        """
        self._data_forward = LMDataset(
            filepaths, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self._data_reverse = LMDataset(
            filepaths, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        for X, Xr in zip(
            _get_batch(self._data_forward.get_sentence(), batch_size,
                      num_steps, max_word_length),
            _get_batch(self._data_reverse.get_sentence(), batch_size,
                      num_steps, max_word_length)
            ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X


class BidirectionalPolyglotLMDataset(BidirectionalLMDataset):
    def __init__(self, filepaths, vocab, test=False, shuffle_on_load=False):
        """
        bidirectional version of PolyglotLMDataset
        """
        print("Creating forward PolyglotLMDataset")
        self._data_forward = PolyglotLMDataset(
            filepaths, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        print("Creating backward PolyglotLMDataset")
        self._data_reverse = PolyglotLMDataset(
            filepaths, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)


class InvalidNumberOfCharacters(Exception):
    pass

