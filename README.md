A fork of the [AllenNLP](http://www.allennlp.org/) research library with extensions for training polyglot models.

## Overview

* bilm-tf: based on the [bilm-tf](https://github.com/allenai/bilm-tf) library, for training multilingual LMs
* allennlp: based on the [allennlp](https://github.com/allenai/allennlp) library, for using contextual word embeddings from multilingual LMs.

## Installation

`bilm-tf` and `allennlp` have different requirements.
You will need a python 3.5 environment with tensorflow version 1.2 and h5py to use bilm-tf to train multilingual language models.
You will need a python 3.6 environment with PyTorch version >=0.4.1 to use the multilingual language models for contextual embeddings in AllenNLP models.
See [the original bilm-tf README](https://github.com/allenai/bilm-tf/blob/master/README.md) and [the original allennlp README](https://github.com/allenai/allennlp/blob/master/README.md) (install from source) for installation details.

## Training language models with Rosita

You can train a polyglot language model with the following command (in the python 3.5 + tensorflow environment):

```
python bilm-tf/bin/train_elmo_poly.py --save_dir [path/to/model/dir] --vocab_file [path/to/vocab.txt] --train_paths [multiple/paths/to/files_*.txt]
```

You can either pass `--n_train_tokens [X]` or edit the default value for `n_train_tokens` in `bilm-tf/bin/train_elmo_poly.py` to reflect the number of tokens in your corpus.

Optionally, you can specify a gpu to train with using the flag `--gpu k`.

The vocabulary file should have one word per line, starting with the special tokens `<S>`, `</S>` and `<UNK>` and sorted by descending frequency.
Non-special words should be prefixed with a language code, e.g. `eng:example`.
You can produce a vocab file from your training data with the  `build_vocab.py` script:

```
python bilm-tf/build_vocab.py [/path/to/corpus.txt] [paths/to/additional/textfiles]
```

## Dumping weights

Once you have trained a LM, you can save its parameters as an HDF5 file to be read by AllenNLP.

```
python bilm-tf/bin/dump_weights.py --save_dir [path/to/training/output] --outfile [path to save weights, e.g. save_dir/weights.hdf5] --gpu [id]
```

## Running with AllenNLP

Once you've installed our edited AllenNLP from source, you can run the command-line interface with `bin/allennlp` (in the python 3.6 + PyTorch environment).

An example training configuration is provided for a Universal Dependencies syntactic parser. Once you have an Arabic-English LM, the parser can be trained with the command:

```
allennlp train training_config/ud-ara-eng_elmo-ara-eng.json --serialization-dir models/rosita-test-ara-eng
```

## Citing

If you use our method, please cite the paper [Polyglot Contextual Representations Improve Crosslingual Transfer](https://arxiv.org/abs/1902.09697).

```
@inproceedings{Mulcaire2019Polyglot,
  title={Polyglot Contextual Representations Improve Crosslingual Transfer},
  author={Phoebe Mulcaire and Jungo Kasai and Noah A. Smith},
  booktitle={Proc.\ of NAACL-HLT},
  year={2019},
  Eprint = {arXiv:1902.09697},
}
```

You should also cite [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).

```
@inproceedings{Gardner2017AllenNLP,
  title={AllenNLP: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2017},
  Eprint = {arXiv:1803.07640},
}
```
