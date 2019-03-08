import argparse
import os
import numpy as np

from bilm.training_new import train, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import BidirectionalLMDataset, BidirectionalPolyglotLMDataset

import IPython as ipy

def main(args):
    print("\n\n ------- RUNNING THE WORD+CHAR VERSION, NOT THE DEFAULT ------- \n\n\n")
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
    #                 2558346 (for eng dev/)
    #                57029976 (for arabic train/)
    #                70546273 (for english .tok train/)
    #                76386340 (for chineseS .tok train/)
    #                64928316 (for chineseT .tok train/)
    #               146932613 (for english + chineseS .tok train/)
    #               135474589 (for english + chineseT .tok train/)
    #               127576249 (for english + arabic .tok train/)
    n_train_tokens = 127576249

    """
    again, this is the experimental version.
    """

    options = {
     'bidirectional': True,

     'word_emb': {
         'file':args.word_emb,
         'output_dim': 300,
         'vocab': os.path.abspath(args.vocab_file),
         'word_map':os.path.abspath(os.path.join(args.save_dir,"word_vocab.txt"))
     },
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
         'n_characters': vocab.n_chars,
         'n_highway': 2,
         'output_dim':212,
         'char_map':os.path.abspath(os.path.join(args.save_dir,"char_vocab.txt"))
     },
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 2048,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
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

    train(options, data, None, n_gpus, tf_save_dir, tf_log_dir, restart_ckpt_file=None) #change restart_ckpt_file to a checkpoint filename to continue training from that checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_paths', nargs='+', help='Filenames for train files')
    parser.add_argument('--word_emb', help='File with pretrained embeddings, _unprefixed_')
    parser.add_argument('--gpu', nargs='+', help='GPU id')
    args = parser.parse_args()
    main(args)

