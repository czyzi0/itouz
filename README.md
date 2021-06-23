# ITOUZ - projekt

Repozytorium zawierające implementację projektu na przedmiot ITOUZ (Inteligentne Techniki Obliczeniowe) realizowanego w ramach studiów magisterskich.

## Uruchamianie

Najpierw należy zainstalować potrzebne pakiety. Polecam stosować oddzielne środowisko, np. virtualenv albo conda. (Projekt był implementowany z wykorzystaniem Python=3.8). Krok ten można pominąć w przypadku korzystania z plików wykonywalnych.
```
pip install -r requirements.txt
```

W następnej kolejności należy pobrać dane ze strony https://deepai.org/dataset/timit oraz wypakować je na dysku.

Kolejnym krokiem jest ekstrakcja cech. Wykonywana jest ona skryptem `extract_feats.py`:
```
python extract_feats.py --input ${ŚCIEŻKA_DO_DANYCH} --output ${ŚCIEŻKA_DOCELOWA}
# na przykład
python extract_feats.py --input data/timit/data/TRAIN --output data/train
python extract_feats.py --input data/timit/data/TEST --output data/test
```
lub za pomocą pliku binarnego `extract_feats.exe` o takich samych parametrach.

Dwa poprzednie kroki (przygotanie danych) można zastąpić pobraniem przygotowanych już danych z https://drive.google.com/file/d/1FDr0EdHKP9G0fdianrGlZPqVzCciwr_8/view?usp=sharing

Uruchamianie eksperymentu odbywa się przy pomocy skryptu `run_experiment.py`:
```
>>> python run_experiment.py -h
usage: run_experiment.py [-h] --train TRAIN --test TEST [--rnn_size RNN_SIZE] [--epochs EPOCHS] [--output OUTPUT]

Train and evalute model

optional arguments:
  -h, --help           show this help message and exit
  --train TRAIN        path to directory with prepared training data
  --test TEST          path to directory with prepared testing data
  --rnn_size RNN_SIZE  size of the RNN layer in the model
  --epochs EPOCHS      number of epochs to train the model for
  --output OUTPUT      output file where classification results for test set will be saved
```
Przykład:
```
python run_experiment.py --train data/train/ --test data/test/ --rnn_size 128 --epochs 45
```
lub za pomocą pliku wykonywalnego `run_experiment.exe` o takich samych parametrach wywołania.

UWAGA: Ze względu na rozmiar plików binarnych i ograniczenie całości projektu do 20MB, całość projektu można pobrać z publicznego repozytorium https://github.com/czyzi0/itouz

## Uruchamianie z plików binarnych

Należy pominąć krok związny z tworzeniem środowiska. 

## Rozkład plików

`auxiliary/` - dodatkowe pliki, takie jak rysunki załączone do raportu, czy przykładowe wyniki zbioru testowego.

`logs/` - wymagane pliki-logi z uruchomienia programu; załączone zostały pliki logi z każdego raportowanego treningu oraz z ekstrakcji cech.

`*.py` - pliki źródłowe projektu.

`*.exe` - pliki wykonywalne przygotowane dla skryptów.

`README.md` - ten plik.

`report.pdf` - raport końcowy z projektu.

