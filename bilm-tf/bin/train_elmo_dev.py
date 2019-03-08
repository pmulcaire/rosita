import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import BidirectionalLMDataset, BidirectionalPolyglotLMDataset

import IPython as ipy

def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_files, max_word_length=50, polyglot=args.polyglot)

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
            'embedding': {'dim': 16},
            'filters': [[1, 32],
                        [2, 32],
                        [3, 64],
                        [4, 128],
                        [5, 256],
                        [6, 512],
                        [7, 1024]],
            'max_characters_per_token': 50,
            'n_characters': 261,
            'n_highway': 2},
    
        'dropout': 0.1,
        
        'lstm': {
            'cell_clip': 3,
            'dim': 2048,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 256,
            'use_skip_connections': True},
        
        'all_clip_norm_val': 10.0,
        
        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    train_prefix = args.train_paths
    dev_prefix = args.dev_paths

    if args.polyglot:
        data = BidirectionalPolyglotLMDataset(train_paths, vocab, **kwargs,
                                              test=False,shuffle_on_load=True)
        dev_data = BidirectionalPolyglotLMDataset(dev_prefix,vocab,test=False,
                                                  shuffle_on_load=True)
    else:
        data = BidirectionalLMDataset(train_prefix, vocab, test=False,
                                      shuffle_on_load=True)
        dev_data = BidirectionalLMDataset(dev_prefix, vocab, test=False, 
                                          shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, dev_data, n_gpus,
          tf_save_dir, tf_log_dir,
          restart_ckpt_file=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_files', nargs='+', help='Vocabulary file')
    parser.add_argument('--train_paths', nargs='+', help='Filenames for train files')
    parser.add_argument('--dev_paths', nargs='+', help='Filenames for dev files')
    parser.add_argument('--gpu', nargs='+', help='GPU id')
    parser.add_argument('--polyglot', action='store_true', help='train a polyglot model')

    args = parser.parse_args()
    main(args)

