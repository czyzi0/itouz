import argparse
import os
import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


TOKENS = [
    'h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b',
    'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
    'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy',
    'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau',
    'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v',
    'w', 'y', 'z', 'zh']


def train_test_split(data_dir, test_size=0.1):
    ids = []
    with open(data_dir / 'metadata.tsv', 'r') as fh:
        for line in fh:
            id_, *_ = line.strip().split('\t')
            ids.append(id_)
    ids = np.array(ids)

    test_size = int(test_size * len(ids))

    idx = np.arange(len(ids))
    np.random.shuffle(idx)

    idx_test = np.sort(idx[:test_size])
    idx_train = np.sort(idx[test_size:])

    ids_test = ids[idx_test]
    ids_train = ids[idx_train]

    return list(ids_train), list(ids_test)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, ids=None):
        self.data_dir = data_dir

        self.data = []
        self.mel_lens = []

        self.tok2id = {t: i for i, t in enumerate(TOKENS)}

        with open(data_dir / 'metadata.tsv', 'r') as fh:
            for line in fh:
                id_, _, _, labels, mel_len = line.strip().split('\t')
                if ids is None or id_ in ids:
                    mel_len = int(mel_len)
                    self.data.append((id_, labels))
                    self.mel_lens.append(mel_len)

    def __getitem__(self, index):
        id_, labels = self.data[index]

        mel = np.load(self.data_dir / 'mels' / f'{id_}.npy')
        labels = np.array([self.tok2id[t] for t in labels.split()])

        return mel, labels

    def __len__(self):
        return len(self.data)


class MyCollate:

    def __init__(self, data_dir):
        self.padding_frame = np.load(data_dir / 'min_frame.npy').reshape(-1, 1)

    def __call__(self, batch):
        mels = collate_mels([x[0] for x in batch], self.padding_frame)
        labels = collate_labels([x[1] for x in batch])

        return torch.FloatTensor(mels), torch.LongTensor(labels)


def collate_mels(mels, padding_frame):
    max_len = max(m.shape[-1] for m in mels)

    mels_ = []
    for mel in mels:
        mel_len = mel.shape[1]
        mel_ = np.zeros((mel.shape[0], max_len))
        mel_[:, :mel.shape[1]] = mel
        mel_[:, mel.shape[1]:] = padding_frame
        mels_.append(mel_)
    mels = np.stack(mels_)

    return mels


def collate_labels(labels):
    max_len = max(len(l) for l in labels)
    # We can pad with `0` thanks to `h#` being labeled as `0`
    labels = np.stack([np.pad(l, (0, max_len - len(l)), mode='constant') for l in labels])

    return labels


class BinnedLengthSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, lens, batch_size, shuffle=True):
        super().__init__(dataset)

        self.shuffle = shuffle

        def _chunked(array, size):
            for i in range(0, len(array), size):
                yield array[i:i + size]

        sorted_indices = np.array(lens).argsort()
        self.batches = list(_chunked(sorted_indices, batch_size))

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i] for i in np.random.permutation(len(self.batches)))
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class MyModel(pl.LightningModule):

    def __init__(self, rnn_size, n_layers, n_classes):
        super().__init__()

        self.rnn = nn.RNN(80, rnn_size, num_layers=n_layers, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(rnn_size, n_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, mels):
        x = mels.transpose(1, 2)

        x, _ = self.rnn(x)
        x = self.linear(x)

        x = x.transpose(1, 2)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('loss/train', loss)
        self.log('acc/train_step', self.train_acc(F.softmax(outputs, dim=1), targets))
        return loss

    def training_epoch_end(self, outs):
        self.log('acc/train_epoch', self.train_acc.compute())

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('loss/val', loss)
        self.log('acc/val', self.val_acc(F.softmax(outputs, dim=1), targets), on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        inputs, targets = test_batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('loss/test', loss)
        self.log('acc/test', self.test_acc(F.softmax(outputs, dim=1), targets), on_epoch=True)

    def details(self):
        n_params_rnn = sum([
            np.prod(p.size()) for p in self.rnn.parameters() if p.requires_grad])
        n_params_linear = sum([
            np.prod(p.size()) for p in self.linear.parameters() if p.requires_grad])
        n_params = n_params_rnn + n_params_linear

        details = (
            f'----------------MyModel------------------\n'
            f'module                       # of params \n'
            f'-----------------------------------------\n'
            f'rnn                          {n_params_rnn:,}\n'
            f'linear                       {n_params_linear:,}\n'
            f'-----------------------------------------\n'
            f'TOTAL                        {n_params:,}\n')
        return details


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--train', type=pathlib.Path, required=True, help='')
    parser.add_argument('--test', type=pathlib.Path, required=True, help='')

    return parser.parse_args()


def main(train_dir, test_dir):
    np.random.seed(1234)
    torch.manual_seed(1234)

    device = torch.device('cpu')

    train_ids, val_ids = train_test_split(train_dir)

    train_set = MyDataset(train_dir, train_ids)
    train_loader = DataLoader(
        train_set,
        batch_sampler=BinnedLengthSampler(train_set, train_set.mel_lens, batch_size=21),
        collate_fn=MyCollate(train_dir),
        num_workers=os.cpu_count())

    val_set = MyDataset(train_dir, val_ids)
    val_loader = DataLoader(
        val_set,
        batch_sampler=BinnedLengthSampler(val_set, val_set.mel_lens, batch_size=21),
        collate_fn=MyCollate(train_dir),
        num_workers=os.cpu_count())

    test_set = MyDataset(test_dir)
    test_loader = DataLoader(
        test_set,
        batch_sampler=BinnedLengthSampler(test_set, test_set.mel_lens, batch_size=1),
        collate_fn=MyCollate(test_dir),
        num_workers=os.cpu_count())

    print('Prepared data:')
    print(f'Train - samples: {len(train_set)} - batches: {len(train_loader)}')
    print(f'Val - samples: {len(val_set)} - batches: {len(val_loader)}')
    print(f'Test - samples: {len(test_set)} - batches: {len(test_loader)}')
    print()

    model = MyModel(rnn_size=256, n_layers=4, n_classes=len(TOKENS))
    model.train()
    checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=3, mode='min')
    print('Prepared model:')
    print(model.details())

    trainer = pl.Trainer(gpus=1, max_steps=25000, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args.train, args.test)
