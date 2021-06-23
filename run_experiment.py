"""Main script that performs model training and evaluation."""

import argparse
import pathlib

import numpy as np

import nn
from data import prepare_dataset, train_test_split, TOKENS
from utils import init_logging, close_logging, my_print


N_MELS = 80


def parse_args():
    """Function for parsing arguments for the script"""
    parser = argparse.ArgumentParser(description='Train and evalute model')

    parser.add_argument(
        '--train', type=pathlib.Path, required=True,
        help='path to directory with prepared training data')
    parser.add_argument(
        '--test', type=pathlib.Path, required=True,
        help='path to directory with prepared testing data')
    parser.add_argument(
        '--rnn_size', type=int, default=128,
        help='size of the RNN layer in the model')
    parser.add_argument(
        '--epochs', type=int, default=15,
        help='number of epochs to train the model for')
    parser.add_argument(
        '--output', type=pathlib.Path, default=pathlib.Path('results.tsv'),
        help='output file where classification results for test set will be saved')

    return parser.parse_args()


def main(train_dir, test_dir, rnn_size, epochs, output_fp):
    """Main function of the script that does the job."""
    np.random.seed(1234)

    my_print('Preparing data...', end='', flush=True)
    train_ids, val_ids = train_test_split(train_dir)
    train_x, train_y = prepare_dataset(train_dir, batch_size=21, ids=train_ids)
    val_x, val_y = prepare_dataset(train_dir, batch_size=21, ids=val_ids)
    test_x, test_y = prepare_dataset(test_dir, batch_size=1)
    my_print('DONE')
    my_print(f'Training - Batches: {len(train_x)} - Samples: {len(train_x) * train_x[0].shape[0]}')
    my_print(f'Validation - Batches: {len(val_x)} - Samples: {len(val_x) * val_x[0].shape[0]}')
    my_print(f'Testing - Batches: {len(test_x)} - Samples: {len(test_x) * test_x[0].shape[0]}')
    my_print()

    my_print('Preparing model...', end='', flush=True)
    model = nn.Model(
        layers=[
            nn.RNN(N_MELS, rnn_size),
            #nn.RNN(rnn_size, rnn_size),
            nn.Linear(rnn_size, len(TOKENS)),
            nn.Softmax(),
        ],
        loss=nn.CrossEntropyLoss(),
        train_metrics=[nn.Accuracy()],
        val_metrics=[nn.Accuracy()]
    )
    my_print('DONE')
    my_print(model)
    my_print()

    my_print('Started training:')
    model.train(train_x, train_y, val_x, val_y, lr=0.0001, momentum=0.8, epochs=epochs)
    my_print()

    my_print('Testing:', end='')
    metric = nn.Accuracy()
    with open(output_fp, 'w') as fh:
        for x, y_true in zip(test_x, test_y):
            y_pred = model.predict(x)

            metric.log(y_true, y_pred)

            y_pred = y_pred.argmax(axis=2)[0]
            y_true = y_true.argmax(axis=2)[0]

            y_pred = ' '.join([TOKENS[i] for i in y_pred])
            y_true = ' '.join([TOKENS[i] for i in y_true])

            fh.write(f'{y_pred}\t{y_true}\n')

    my_print(f' - {metric}: {metric.calc():.4f}')


if __name__ == '__main__':
    init_logging('run_experiment.log.txt')
    args = parse_args()
    main(args.train, args.test, args.rnn_size, args.epochs, args.output)
    close_logging()
