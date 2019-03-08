
import argparse

from bilm.training import test, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import LMDataset, BidirectionalLMDataset, BidirectionalPolyglotLMDataset

def main(args):

    if args.gpu is not None:
        n_gpus = len(args.gpu)
        set_gpu(args.gpu)
    else:
        n_gpus = 0

    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None

    if options.get('polyglot'):
        vocab = load_vocab(args.vocab_files, max_word_length, polyglot=True)
    else:
        vocab = load_vocab(args.vocab_files, max_word_length)

    test_prefix = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('polyglot'):
        data = BidirectionalPolyglotLMDataset(test_prefix, vocab, **kwargs)
    elif options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_files', nargs='+', help='Vocabulary file')
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--batch_size',
        type=int, default=128,
        help='Batch size')
    parser.add_argument('--gpu', nargs='+', help='GPU id')
    args = parser.parse_args()
    main(args)

