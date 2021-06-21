import numpy as np


# 'h#' is first on the list on purpose, it means silence, it is used for padding
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


def prepare_dataset(data_dir, batch_size, ids=None):
    # Load data
    mels, labels_, mel_lens = [], [], []
    with open(data_dir / 'metadata.tsv', 'r') as fh:
        for line in fh:
            id_, _, _, labels, mel_len = line.strip().split('\t')
            if ids is None or id_ in ids:
                mels.append(np.load(data_dir / 'mels' / f'{id_}.npy'))
                labels_.append(np.array(labels.split()))
                mel_lens.append(int(mel_len))
    # Load padding frame
    padding_frame = np.load(data_dir / 'min_frame.npy').reshape(-1, 1)

    # Bin mels of similar length into batches
    sorted_indices = np.array(mel_lens).argsort()
    batches = list(_chunked(sorted_indices, batch_size))

    # Prepare inputs (padded with min frame)
    x = [
        _collate_mels([mels[i] for i in batch], padding_frame).transpose(0, 2, 1)
        for batch in batches]
    # Prepare targets (padded with 'h#' and one-hot encoded)
    y = [
        _collate_labels([labels_[i] for i in batch])
        for batch in batches]

    return x, y


def _chunked(array, size):
    for i in range(0, len(array), size):
        yield array[i:i + size]


def _collate_mels(mels, padding_frame):
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


def _collate_labels(labels):
    max_len = max(len(l) for l in labels)

    labels_ohe = []
    for l in labels:
        l = np.pad(l, (0, max_len - len(l)), mode='constant', constant_values='h#')
        l_ohe = np.zeros((max_len, len(TOKENS)))
        for i, l_ in enumerate(l):
            l_ohe[i, TOKENS.index(l_)] = 1
        labels_ohe.append(l_ohe)
    labels_ohe = np.stack(labels_ohe)

    return labels_ohe
