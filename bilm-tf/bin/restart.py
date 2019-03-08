import argparse
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import LMDataset, BidirectionalLMDataset, BidirectionalPolyglotLMDataset

def main(args):

    if args.gpu is not None:
        n_gpus = len(args.gpu)
        set_gpu(args.gpu)
    else:
        n_gpus = 0

    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None

    if 'polyglot' in options:
        vocab = load_vocab(args.vocab_file, max_word_length, polyglot=True)
    else:
        vocab = load_vocab(args.vocab_file, max_word_length)

    train_paths = args.train_paths

    kwargs = {
        'test': False,
        'shuffle_on_load': True,
    }

    if 'polyglot' in options:
        data = BidirectionalPolyglotLMDataset(train_paths, vocab, **kwargs)
    elif options.get('bidirectional'):
        data = BidirectionalLMDataset(train_paths, vocab, **kwargs)
    else:
        data = LMDataset(train_paths, vocab, **kwargs)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    # set optional inputs
    if args.n_train_tokens > 0:
        options['n_train_tokens'] = args.n_train_tokens
    if args.n_epochs > 0:
        options['n_epochs'] = args.n_epochs
    if args.batch_size > 0:
        options['batch_size'] = args.batch_size

    train(options, data, None, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary files')
    parser.add_argument('--train_paths', nargs='+', help='Prefix for train files')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--n_train_tokens', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=0)
    parser.add_argument('--gpu', nargs='+', help='GPU id(s)')

    args = parser.parse_args()
    main(args)

