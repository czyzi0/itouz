import argparse
import pathlib
import shutil

import librosa
import numpy as np
from scipy.io.wavfile import read


SAMPLING_RATE = 16000
MAX_WAV_VALUE = 32678
N_MELS = 80

FILTER_LEN = 1024
HOP_LEN = 160
WIN_LEN = 1024

MEL_FMIN = 0.0
MEL_FMAX = 8000.0


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input', type=pathlib.Path, required=True, help='')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='')

    return parser.parse_args()


def load_wav(fp):
    sr, wav = read(str(fp))
    assert sr == SAMPLING_RATE
    wav = np.clip(wav / MAX_WAV_VALUE, -1, 1)
    return wav


def convert_wav2mel(wav):
    spectrogram = librosa.stft(y=wav, n_fft=FILTER_LEN, hop_length=HOP_LEN, win_length=WIN_LEN)
    spectrogram = np.abs(spectrogram)
    mel = librosa.feature.melspectrogram(
        S=spectrogram, sr=SAMPLING_RATE, n_fft=FILTER_LEN, n_mels=N_MELS,
        fmin=MEL_FMIN, fmax=MEL_FMAX)
    mel = np.clip(mel, 1e-5, None)
    mel = np.log(mel)
    return mel


def main(input_dir, output_dir):
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'wavs').mkdir(exist_ok=True)
    (output_dir / 'mels').mkdir(exist_ok=True)

    lines, i = [], 0
    for dialect_dir in sorted(input_dir.glob('*')):
        for speaker_dir in sorted(dialect_dir.glob('*')):
            for text_fp in sorted(speaker_dir.glob('*.TXT')):
                # Prepare id
                id_ = f'{dialect_dir.stem}_{speaker_dir.stem}_{text_fp.stem}'
                # Load and copy wav
                wav_fp = speaker_dir / f'{text_fp.stem}.WAV.wav'
                wav = load_wav(wav_fp)
                shutil.copy(wav_fp, output_dir / 'wavs' / f'{id_}.wav')
                # Extract features and save them
                mel = convert_wav2mel(wav)
                np.save(output_dir / 'mels' / f'{id_}.npy', mel)
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
                tscp, tscp_ext = ' '.join(tscp), ' '.join(tscp_ext)
                # Read text
                with open(text_fp, 'r') as fh:
                    text = ' '.join(fh.read().split()[2:])

                i += 1
                lines.append(f'{id_}\t{text}\t{tscp}\t{tscp_ext}\t{mel.shape[1]}')

                print(f'\rPrepared {i} files', end='')
    print()

    metadata_fp = output_dir / 'metadata.tsv'
    with open(metadata_fp, 'w') as fh:
        for line in lines:
            fh.write(f'{line}\n')
    print(f'Saved metadata to {metadata_fp}')


if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.output)
