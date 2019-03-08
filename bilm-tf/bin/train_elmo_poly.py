import argparse
import os
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import BidirectionalLMDataset, BidirectionalPolyglotLMDataset

import IPython as ipy

def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, max_word_length=50, polyglot=True)
    vocab.save_vocab(args.save_dir)

    # define the options
    batch_size = 128  # batch size for each GPU

    if args.gpu is not None:
        n_gpus = len(args.gpu)
        set_gpu(args.gpu)
    else:
        n_gpus = 0

    # number of tokens in training data
    #                768648884 (for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
        'bidirectional': True,

        'char_cnn': {
            'activation': 'relu',
            'char_map': os.path.abspath(os.path.join(args.save_dir,'char_vocab.txt')),
            'embedding': {'dim': 16},
            'filters': [[1, 32],
                        [2, 32],
                        [3, 64],
                        [4, 128],
                        [5, 256],
                        [6, 512],
                        [7, 1024]],
            'max_characters_per_token': 50,
            'n_characters': vocab.n_chars,
            'n_highway': 2
        },

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 2048,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 256,
            'use_skip_connections': True
        },
    
        'all_clip_norm_val': 10.0,
    
        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    train_paths = args.train_paths
    data = BidirectionalPolyglotLMDataset(train_paths, vocab, test=False,
                                          shuffle_on_load=True)
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    train(options, data, None, n_gpus,
          tf_save_dir, tf_log_dir,
          restart_ckpt_file=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_paths', nargs='+', help='Filenames for train files')
    parser.add_argument('--gpu', nargs='+', help='GPU id')

    args = parser.parse_args()
    main(args)

