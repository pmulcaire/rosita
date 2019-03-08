"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
    archive_file          path to an archived trained model
    input_file            path to the file containing the evaluation data

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file to save metrics
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any, Iterable, TextIO
import argparse, os
import logging
import json
from collections import defaultdict

import torch

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data import conllu_parse
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file 
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset')

        subparser.add_argument('archive_file', type=str, help='path to an archived trained model')

        subparser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')
        subparser.add_argument('--evaluation_file', type=str, help='path to the evaluation file (for UD)')
        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--outfile', type=str, help='outfile name', default='predicted.conll')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')
        subparser.add_argument('--print_predictions',
                           help='print CoNLL files containing the predicted and labelings')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    _warned_tqdm_ignores_underscores = False
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
        for batch in generator_tqdm:
            batch = util.move_to_device(batch, cuda_device)
            model(**batch)
            metrics = model.get_metrics()
            if (not _warned_tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                _warned_tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        return model.get_metrics(reset=True)

def evaluate_predict(model: Model,
                     instances: Iterable[Instance],
                     data_iterator: DataIterator,
                     cuda_device: int,
                     predict_file: TextIO,
                     gold_file: TextIO) -> Dict[str, Any]:
    model.eval() #sets the model to evaluation mode--no dropout, batchnorm, other stuff?
    ## Unfortunately, we cannot use the data_iterator because it converts batches to tensors....
    iterator = data_iterator.feed_predictions(instances,
                             num_epochs=1,
                             shuffle=False,
                             sorting=True)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

    print("setting up conll output")
    for batch in generator_tqdm:
        output_batch = model.forward_on_instances(batch.instances)
        assert len(output_batch) == len(list(batch.instances))
        for output, instance in zip(output_batch, batch.instances):
            predicted_tags = output['tags']
            gold_tags = instance.fields['tags'].labels
            tokens = instance.fields['tokens'].tokens
            words = [t.text for t in tokens]
            pred_indices = instance.fields['verb_indicator'].labels
            
            if 1 in pred_indices:
                pred_idx = pred_indices.index(1)
                write_to_conll_eval_file(predict_file,
                                     gold_file,
                                     pred_idx,
                                     words,
                                     predicted_tags,
                                     gold_tags)
    print("printed conll output")

def evaluate_predict_ud(model: Model,
                     instances: Iterable[Instance],
                     data_iterator: DataIterator,
                     cuda_device: int,
                     input_file: str,
                     predict_file: TextIO) -> Dict[str, Any]:
    model.eval() #sets the model to evaluation mode--no dropout, batchnorm, other stuff?
    ## Unfortunately, we cannot use the data_iterator because it converts batches to tensors....
    iterator = data_iterator.feed_predictions(instances,
                             num_epochs=1,
                             shuffle=False,
                             sorting=False)
    print('We do not sort UD instances to preserve sentence ordering')
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    with open(input_file) as fin:
        output_data = fin.read()
    output_conllu = conllu_parse(output_data) 

    print("setting up conllu output")
    sent_idx = 0
    all_output = []
    for batch in generator_tqdm:
        output_batch = model.forward_on_instances(batch.instances)
        all_output.extend(output_batch)
    for output in all_output:
        sent_conllu = output_conllu[sent_idx]
        token_id = 0
        count = 0
        for token_conllu_id in range(len(sent_conllu)):
            if sent_conllu[token_conllu_id]['id'] == token_id + 1:## if not, multi-word expression, so skip. +1 for staring from 1 (0=ROOT)
                head = output['predicted_heads'][token_id]
                deprel = output['predicted_dependencies'][token_id]
                sent_conllu[token_conllu_id]['head'] = head
                sent_conllu[token_conllu_id]['deprel'] = deprel
                #sent_conllu[token_conllu_id]['deps'][0] = (deprel, head)
                count += 1
                token_id += 1
        assert (count == len(output['words']))
        ## check we exhaust all words
        sent_idx += 1
    for sent_conllu in output_conllu:
        predict_file.write(sent_conllu.serialize())
    print("printed conllu output")
    print(sent_idx)
    print(len(output_conllu))
    assert sent_idx == len(output_conllu)

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    torch.cuda.set_device(args.cuda_device)
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    if args.print_predictions=='srl':
        directory = os.path.dirname(args.archive_file)
        predict_filename = args.print_predictions
        predict_file = open(os.path.join(directory, args.outfile), 'w')
        gold_file = open(os.path.join(directory, 'gold.conll'), 'w')
        evaluate_predict(model, instances, iterator, args.cuda_device, predict_file, gold_file)
        predict_file.close()
        gold_file.close()
        return True
    elif args.print_predictions=='ud':
        directory = os.path.dirname(args.archive_file)
        predict_filename = args.print_predictions
        predict_file = open(os.path.join(directory, args.outfile), 'w')
        evaluate_predict_ud(model, instances, iterator, args.cuda_device, args.evaluation_file, predict_file)
        predict_file.close()
        return True
    else:
        metrics = evaluate(model, instances, iterator, args.cuda_device)
        logger.info("Finished evaluating.")
        logger.info("Metrics:")
        for key, metric in metrics.items():
            logger.info("%s: %s", key, metric)

        output_file = args.output_file
        if output_file:
            with open(output_file, "w") as file:
                json.dump(metrics, file, indent=4)
        return metrics
