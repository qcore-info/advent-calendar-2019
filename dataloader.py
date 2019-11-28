import os
import random

import wfdb
import numpy as np
from imblearn.datasets import make_imbalance
from scipy.signal import butter, lfilter

from qore_sdk.utils import sliding_window


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PKG_ROOT, 'dataset')


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


reset_seed(42)


def download():

    print('Downloading data...')
    download_dir = os.path.join(DATASET_ROOT, 'download')
    wfdb.dl_database('mitdb', dl_dir=download_dir)


class BaseECGDatasetPreprocessor(object):

    def __init__(
            self,
            dataset_root,
            window_size=720,  # 2 seconds
    ):
        self.dataset_root = dataset_root
        self.download_dir = os.path.join(self.dataset_root, 'download')
        self.window_size = window_size
        self.sample_rate = 360.
        # split list
        self.train_record_list = [
            '101', '106', '108', '109', '112', '115', '116', '118', '119', '122',
            '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
        ]
        self.test_record_list = [
            '100', '103', '105', '111', '113', '117', '121', '123', '200', '210',
            '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
        ]
        # annotation
        self.labels = ['N', 'V']
        self.valid_symbols = ['N', 'L', 'R', 'e', 'j', 'V', 'E']
        self.label_map = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
            'V': 'V', 'E': 'V'
        }

    def _load_data(
            self,
            base_record,
            channel=0  # [0, 1]
    ):
        record_name = os.path.join(self.download_dir, str(base_record))
        # read dat file
        signals, fields = wfdb.rdsamp(record_name)
        assert fields['fs'] == self.sample_rate
        # read annotation file
        annotation = wfdb.rdann(record_name, 'atr')
        symbols = annotation.symbol
        positions = annotation.sample
        return signals[:, channel], symbols, positions

    def _normalize_signal(
            self,
            signal,
            method='std'
    ):
        if method == 'minmax':
            # Min-Max scaling
            min_val = np.min(signal)
            max_val = np.max(signal)
            return (signal - min_val) / (max_val - min_val)
        elif method == 'std':
            # Zero mean and unit variance
            signal = (signal - np.mean(signal)) / np.std(signal)
            return signal
        else:
            raise ValueError("Invalid method: {}".format(method))

    def _segment_data(
            self,
            signal,
            symbols,
            positions
    ):
        X = []
        y = []
        sig_len = len(signal)
        for i in range(len(symbols)):
            start = positions[i] - self.window_size // 2
            end = positions[i] + self.window_size // 2
            if symbols[i] in self.valid_symbols and start >= 0 and end <= sig_len:
                segment = signal[start:end]
                assert len(segment) == self.window_size, "Invalid length"
                X.append(segment)
                y.append(self.labels.index(self.label_map[symbols[i]]))
        return np.array(X), np.array(y)

    def preprocess_dataset(
            self,
            normalize=True
    ):
        # preprocess training dataset
        self._preprocess_dataset_core(self.train_record_list, "train", normalize)
        # preprocess test dataset
        self._preprocess_dataset_core(self.test_record_list, "test", normalize)

    def _preprocess_dataset_core(
            self,
            record_list,
            mode="train",
            normalize=True
    ):
        Xs, ys = [], []
        save_dir = os.path.join(self.dataset_root, 'preprocessed', mode)
        for i in range(len(record_list)):
            signal, symbols, positions = self._load_data(record_list[i])
            if normalize:
                signal = self._normalize_signal(signal)
            X, y = self._segment_data(signal, symbols, positions)
            Xs.append(X)
            ys.append(y)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), np.vstack(Xs))
        np.save(os.path.join(save_dir, "y.npy"), np.concatenate(ys))


class DenoiseECGDatasetPreprocessor(BaseECGDatasetPreprocessor):

    def __init__(
            self,
            dataset_root='./',
            window_size=720
    ):
        super(DenoiseECGDatasetPreprocessor, self).__init__(
        dataset_root, window_size)

    def _denoise_signal(
            self,
            signal,
            btype='low',
            cutoff_low=0.2,
            cutoff_high=25.,
            order=5
    ):
        nyquist = self.sample_rate / 2.
        if btype == 'band':
            cut_off = (cutoff_low / nyquist, cutoff_high / nyquist)
        elif btype == 'high':
            cut_off = cutoff_low / nyquist
        elif btype == 'low':
            cut_off = cutoff_high / nyquist
        else:
            return signal
        b, a = butter(order, cut_off, analog=False, btype=btype)
        return lfilter(b, a, signal)

    def _segment_data(
            self,
            signal,
            symbols,
            positions
    ):
        X = []
        y = []
        sig_len = len(signal)
        for i in range(len(symbols)):
            start = positions[i] - self.window_size // 2
            end = positions[i] + self.window_size // 2
            if symbols[i] in self.valid_symbols and start >= 0 and end <= sig_len:
                segment = signal[start:end]
                assert len(segment) == self.window_size, "Invalid length"
                X.append(segment)
                y.append(self.labels.index(self.label_map[symbols[i]]))
        return np.array(X), np.array(y)

    def prepare_dataset(
            self,
            denoise=False,
            normalize=True
    ):
        if not os.path.isdir(self.download_dir):
            self.download_data()

        # prepare training dataset
        self._prepare_dataset_core(self.train_record_list, "train", denoise, normalize)
        # prepare test dataset
        self._prepare_dataset_core(self.test_record_list, "test", denoise, normalize)

    def _prepare_dataset_core(
            self,
            record_list,
            mode="train",
            denoise=False,
            normalize=True
    ):
        Xs, ys = [], []
        save_dir = os.path.join(self.dataset_root, 'preprocessed-denoise', mode)
        for i in range(len(record_list)):
            signal, symbols, positions = self._load_data(record_list[i])
            if denoise:
                signal = self._denoise_signal(signal)
            if normalize:
                signal = self._normalize_signal(signal)
            X, y = self._segment_data(signal, symbols, positions)
            Xs.append(X)
            ys.append(y)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), np.vstack(Xs))
        np.save(os.path.join(save_dir, "y.npy"), np.concatenate(ys))


def make_dataset():

    if not os.path.exists(os.path.join(DATASET_ROOT, 'download')):
        download()

    print('Making dataset...')
    DenoiseECGDatasetPreprocessor(DATASET_ROOT).prepare_dataset(denoise=True)


def load_dataset():

    dataset_path = os.path.join(DATASET_ROOT, 'preprocessed-denoise')

    if not os.path.exists(dataset_path):
        make_dataset()

    X_train = np.load(os.path.join(dataset_path, 'train', 'X.npy'))
    y_train = np.load(os.path.join(dataset_path, 'train', 'y.npy'))
    X_test = np.load(os.path.join(dataset_path, 'test', 'X.npy'))
    y_test = np.load(os.path.join(dataset_path, 'test', 'y.npy'))

    return X_train, X_test, y_train, y_test
