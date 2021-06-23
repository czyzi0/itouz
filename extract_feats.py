"""
Script for extracting features and labels from TIMIT dataset in the state
downloaded from https://deepai.org/dataset/timit
"""

import argparse
import pathlib
import shutil

import librosa
import numpy as np
from scipy.io.wavfile import read

from utils import init_logging, close_logging, my_print


# Paremeters for acoustic features extraction
SAMPLING_RATE = 16000
MAX_WAV_VALUE = 32678
N_MELS = 80

FILTER_LEN = 1024
HOP_LEN = 480
WIN_LEN = 1024

MEL_FMIN = 0.0
MEL_FMAX = 8000.0


def parse_args():
    """Function for parsing arguments for the script"""
    parser = argparse.ArgumentParser(
        description='Extract features and prepare data for further processing')

    parser.add_argument(
        '--input', type=pathlib.Path, required=True,
        help='directory with data for feature extraction, as unzipped from https://deepai.org/dataset/timit')
    parser.add_argument(
        '--output', type=pathlib.Path, required=True,
        help='directory where features will be saved')

    return parser.parse_args()


def load_wav(fp):
    """Function for loading wav file.

    Args:
        fp (pathlib.Path): Wav file path.

    Returns:
        np.array: Wav in the form of numpy array, normalized to (-1, 1).

    """
    sr, wav = read(str(fp))
    assert sr == SAMPLING_RATE
    wav = np.clip(wav / MAX_WAV_VALUE, -1, 1)
    return wav


def convert_wav2mel(wav):
    """Function for converting wav file to acoustic features (mel-spectrogram)

    Args:
        wav (np.array): Signal in the form of numy array, as loaded by
            `load_wav` function.

    Returns:
        np.array: Extracted normalized acoustic features in the form of numpy
            array with shape (n_feats, seq_len).

    """
    spectrogram = librosa.stft(y=wav, n_fft=FILTER_LEN, hop_length=HOP_LEN, win_length=WIN_LEN)
    spectrogram = np.abs(spectrogram)
    mel = librosa.feature.melspectrogram(
        S=spectrogram, sr=SAMPLING_RATE, n_fft=FILTER_LEN, n_mels=N_MELS,
        fmin=MEL_FMIN, fmax=MEL_FMAX)
    mel = np.clip(mel, 1e-5, None)
    mel = np.log(mel)
    mel += 5  # Normalize to so the distribution is more around 0
    return mel


def main(input_dir, output_dir):
    """Main function of the script that does the extraction."""
    output_dir.mkdir(exist_ok=True)

    wavs_dir = output_dir / 'wavs'
    wavs_dir.mkdir(exist_ok=True)
    mels_dir = output_dir / 'mels'
    mels_dir.mkdir(exist_ok=True)

    tokens = set()
    min_frame = np.zeros(N_MELS)
    avg_min_frame = np.mean(min_frame)

    lines, i = [], 0
    for dialect_dir in sorted(input_dir.glob('*')):
        for speaker_dir in sorted(dialect_dir.glob('*')):
            for text_fp in sorted(speaker_dir.glob('*.TXT')):
                # Prepare id
                id_ = f'{dialect_dir.stem}_{speaker_dir.stem}_{text_fp.stem}'

                # Load and copy wav
                wav_fp = speaker_dir / f'{text_fp.stem}.WAV.wav'
                wav = load_wav(wav_fp)
                shutil.copy(wav_fp, wavs_dir / f'{id_}.wav')

                # Extract features and save them
                mel = convert_wav2mel(wav)
                np.save(mels_dir / f'{id_}.npy', mel)

                # Update min frame
                avg_frames = np.mean(mel, axis=0)
                min_idx = np.argmin(avg_frames)
                if avg_frames[min_idx] < avg_min_frame:
                    min_frame = mel[:, min_idx]
                    avg_min_frame = avg_frames[min_idx]

                # Read phonetic transcription and prepare labels for mel frames
                tscp, tscp_ext = [], []
                idx = 0
                with open(speaker_dir / f'{text_fp.stem}.PHN', 'r') as fh:
                    for line in fh:
                        _, end, phoneme = line.strip().split()
                        end = int(end)
                        tscp.append(phoneme)
                        while idx <= end - 0.5 * HOP_LEN:
                            tscp_ext.append(phoneme)
                            idx += HOP_LEN
                        tokens.add(phoneme)
                while mel.shape[1] > len(tscp_ext):
                    tscp_ext.append(tscp_ext[-1])
                tscp, tscp_ext = ' '.join(tscp), ' '.join(tscp_ext)

                # Read text
                with open(text_fp, 'r') as fh:
                    text = ' '.join(fh.read().split()[2:])

                i += 1
                lines.append(f'{id_}\t{text}\t{tscp}\t{tscp_ext}\t{mel.shape[1]}')

                if i % 500 == 0:
                    my_print(f'Prepared {i} files')
    my_print(f'Prepared {i} files')

    metadata_fp = output_dir / 'metadata.tsv'
    with open(metadata_fp, 'w') as fh:
        for line in lines:
            fh.write(f'{line}\n')
    my_print(f'Saved metadata to {metadata_fp}')

    tokens = list(sorted(tokens))
    tokens_fp = output_dir / 'tokens.txt'
    with open(tokens_fp, 'w') as fh:
        fh.write(f'{tokens}\n')
    my_print(f'Saved tokens to {tokens_fp}')

    min_frame_fp = output_dir / 'min_frame.npy'
    np.save(min_frame_fp, min_frame)
    my_print(f'Saved min frame to {min_frame_fp}')


if __name__ == '__main__':
    init_logging('extract_feats.log.txt')
    args = parse_args()
    main(args.input, args.output)
    close_logging()
