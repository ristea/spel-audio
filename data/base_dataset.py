import torch.utils.data
import soundfile as sf
import pandas as pd
import glob
import os
import random
import numpy as np
import librosa as lb
from copy import deepcopy
import pickle
from scipy import signal


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, new_sr=32_000):
        self.sr = sr
        self.new_sr = new_sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sr // 2
        self.res_type = "kaiser_best"

    def transform(self, raw_signal):
        raw_signal = lb.resample(raw_signal, orig_sr=self.sr, target_sr=self.new_sr)
        f, t, spectrogram = signal.stft(raw_signal, fs=self.new_sr, nperseg=512, noverlap=70, window='hamming')
        spectrogram = np.abs(spectrogram) ** 2
        spectrogram = 10 * np.log10(1e-10 + (spectrogram / spectrogram.max()))
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        spectrogram = np.stack((spectrogram, spectrogram, spectrogram))

        return f, t, spectrogram

    def __call__(self, y, color=True, normalize=True, stft=False):
        if self.new_sr is not None:
            y = lb.resample(y, orig_sr=self.sr, target_sr=self.new_sr, res_type=self.res_type)

        if stft is False:
            melspec = lb.feature.melspectrogram(
                y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            )
        else:
            melspec = lb.stft(y, n_fft=1024, hop_length=400, win_length=1024)
            melspec = melspec**2

        melspec = lb.power_to_db(melspec).astype(np.float32)

        melspec_color = self.mono_to_color(melspec) if color else melspec

        if normalize:
            melspec_color = self.normalize(melspec_color)

        return melspec_color

    def mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        X = np.stack([X, X, X], axis=-1)

        # Standardize
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        # Normalize to [0, 255]
        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V

    def normalize(self, image, mean=None, std=None):
        image = image / 255.0
        if mean is not None and std is not None:
            image = (image - mean) / std
        return np.moveaxis(image, 2, 0).astype(np.float32)


def transform(spectrogram, fs=None, config=None):
    spectrogram = np.abs(spectrogram) ** 2
    # spectrogram = pre_process_audio_mel_t(spectrogram, fs, n_mels=self.config['n_mels'])
    spectrogram = np.clip(20 * np.log10(1e-10 + (spectrogram / spectrogram.max())), a_min=-200, a_max=5)
    spectrogram = (spectrogram + 200) / 200

    return spectrogram


def pre_process_audio_mel_t(S, sample_rate=16000, n_mels=128):
    mel_spec = lb.feature.melspectrogram(S=np.abs(S), sr=sample_rate, n_mels=n_mels)
    mel_db = lb.power_to_db(mel_spec, ref=np.max)

    return mel_db


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path, config):
        self.config = config
        self.path = path

        self.mel_spec_computer = MelSpecComputer(sr=config['sr'], fmin=config['f_min'], fmax=config['f_max'],
                                                 n_mels=config['n_mels'], new_sr=config['new_sr'])

        self.sounds_path = list(glob.glob(os.path.join(path, "*.flac")))
        self.tp = pd.read_csv(os.path.join(self.config['labels_path'], "train_tp.csv"))
        self.fp = pd.read_csv(os.path.join(self.config['labels_path'], "train_fp.csv"))

        self.positive_examples = []
        self.prepare_bird_augment()

        if self.config['get_only_tp'] is True:
            self.sounds_path = [os.path.join(path, tmp + ".flac") for tmp in self.tp['recording_id'].values]

        self.sounds_path_spl = list(glob.glob(os.path.join(config['spel_data_path'], "*.pkl")))

    def __getitem__(self, index):
        if self.config['add_spl_data'] is True and index >= len(self.sounds_path):
            data = pickle.load(open(self.sounds_path_spl[index - len(self.sounds_path)], "rb"))
            return data['data'], data['label'], data['data'][0]

        raw_signal, fs = sf.read(self.sounds_path[index])

        tp = deepcopy(self.tp[self.tp['recording_id'] == self.sounds_path[index].split('/')[-1].split('.')[0]])
        labels = None

        if self.config['augment'] is True and self.config['bird_augm'] is True:
            if random.uniform(0, 1) < self.config['bird_chance']:
                alpha = np.random.randint(1, 100) / 100.
                new_sign_idx = np.random.randint(0, len(self.positive_examples) - 1)
                new_sign = self.positive_examples[new_sign_idx]['item']
                pad_beggin = np.random.randint(0, len(raw_signal) - len(new_sign))
                new_sign = np.pad(new_sign, (pad_beggin, len(raw_signal) - len(new_sign) - pad_beggin))
                raw_signal = alpha * raw_signal + (1 - alpha) * new_sign

                labels = np.zeros(24)
                labels[tp['species_id'].values.astype(np.int)] = 1 * alpha
                labels[self.positive_examples[0]['tp'][2]] = 1 - alpha

        tp = tp.reset_index()
        raw_signal, tp = self.crop(raw_signal, tp, fs)

        if self.config['augment'] is True and self.config['noise_augm'] is True:
            if random.uniform(0, 1) < self.config['noise_chance']:
                # Set a target SNR
                target_snr_db = random.randint(20, 40)
                # Calculate signal power and convert to dB
                sig_avg_watts = np.mean(raw_signal ** 2)
                sig_avg_db = 10 * np.log10(sig_avg_watts)
                # Calculate noise according to [2] then convert to watts
                noise_avg_db = sig_avg_db - target_snr_db
                noise_avg_watts = 10 ** (noise_avg_db / 10)
                # Generate an sample of white noise
                mean_noise = 0
                noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), raw_signal.shape)
                # Noise up the original signal
                raw_signal = raw_signal + noise

        spectrogram = self.mel_spec_computer(raw_signal)
        mask = self.get_mask_box(spectrogram, tp)

        if self.config['augment'] is True and self.config['time_augm'] is True:
            if random.uniform(0, 1) < self.config['time_chance']:
                roll_idx = random.randint(0, 6*fs)
                spectrogram = np.roll(spectrogram, roll_idx, axis=-1)
                mask = np.roll(mask, roll_idx, axis=-1)

        if self.config['augment'] is True and self.config['spec_augm']:
            if random.uniform(0, 1) < self.config['spec_chance']:
                spectrogram = self.spec_augment(spectrogram)

        # Compute label
        if labels is None:
            labels = np.zeros(24)
            if len(tp) > 0:
                labels[tp['species_id'].values.astype(np.int)] = 1

        return spectrogram, labels, mask

    def __len__(self):
        if self.config['add_spl_data'] is True:
            return len(self.sounds_path) + len(self.sounds_path_spl)
        return len(self.sounds_path)

    def __repr__(self):
        return self.__class__.__name__

    def spec_augment(self, spectrogram):
        chance = random.uniform(0, 1)

        if chance < 0.33:
            # Temporal augmentation
            start_index = int(random.uniform(0, spectrogram.shape[2]))
            stop_index = int(random.uniform(start_index + 1, spectrogram.shape[2]))
            spectrogram[:, :, start_index:stop_index] = 0

        elif chance > 0.66:
            # Frecv augmentation
            start_index = int(random.uniform(0, spectrogram.shape[1]))
            stop_index = int(random.uniform(start_index + 1, spectrogram.shape[1]))
            spectrogram[:, start_index:stop_index, :] = 0
        else:
            # Both augmentations
            start_index = int(random.uniform(0, spectrogram.shape[2]))
            stop_index = int(random.uniform(start_index + 1, spectrogram.shape[2]))
            spectrogram[:, :, start_index:stop_index] = 0
            start_index = int(random.uniform(0, spectrogram.shape[1]))
            stop_index = int(random.uniform(start_index + 1, spectrogram.shape[1]))
            spectrogram[:, start_index:stop_index, :] = 0

        return spectrogram

    def bird_augment(self, raw_signal, tp):
        new_bird_idx = int(random.uniform(0, len(self.positive_examples)))
        new_bird = self.positive_examples[new_bird_idx]

        new_tp = pd.DataFrame(new_bird['tp']).T
        new_tp['recording_id'] = tp['recording_id'].iloc[0]

        index = random.randint(0, len(raw_signal) - len(new_bird['item'] - 1))
        raw_signal[index:(index+len(new_bird['item']))] = new_bird['item']

        tp = tp.append(new_tp)

        return raw_signal, tp

    def prepare_bird_augment(self):
        sounds_path = [os.path.join(self.path, tmp + ".flac") for tmp in self.tp['recording_id'].values]

        for sound in sounds_path:
            tp = self.tp[self.tp['recording_id'] == sound.split('/')[-1].split('.')[0]]
            tp = tp.reset_index()
            tp = tp.loc[0]

            raw_signal, fs = sf.read(sound)

            idx_t_min = int(tp['t_min'] * fs)
            idx_t_max = int(tp['t_max'] * fs)

            example = raw_signal[idx_t_min:idx_t_max]

            self.positive_examples.append({
                'item': example,
                'tp': tp
            })

    def crop(self, raw_signal, tp, fs):
        tmp_tp = tp.iloc[random.randint(0, len(tp) - 1)]

        idx_t_min = int(tmp_tp['t_min'] * fs)
        idx_t_max = int(tmp_tp['t_max'] * fs)

        while 1:
            start_idx = random.randint(0, len(raw_signal) - self.config['windows_size_fs'] * fs)
            if idx_t_min > start_idx and idx_t_max < start_idx + self.config['windows_size_fs']*fs:
                break

        tp['t_min'] = tp['t_min'] - (start_idx / fs)
        tp['t_max'] = tp['t_max'] - (start_idx / fs)

        tp = tp[tp['t_min'] > 0]
        tp = tp[tp['t_max'] < self.config['windows_size_fs']]

        return raw_signal[start_idx:start_idx + self.config['windows_size_fs'] * fs], tp

    def get_mask_box(self, spect, tps):
        mask = np.zeros(spect.shape[1:])
        tps = tps.reset_index()

        for i in range(0, len(tps)):
            current_tp = tps.loc[i]

            freqs = lb.core.mel_frequencies(fmin=0.0, fmax=self.config['new_sr'] // 2, n_mels=self.config['n_mels'])
            idx_t_min = lb.core.time_to_frames(current_tp['t_min'], self.config['new_sr'], hop_length=512, n_fft=2048)
            idx_t_max = lb.core.time_to_frames(current_tp['t_max'], self.config['new_sr'], hop_length=512, n_fft=2048)

            if idx_t_min < 0:
                idx_t_min = 0

            idx_f_min = np.argmin(abs(freqs - current_tp['f_min']))
            idx_f_max = np.argmin(abs(freqs - current_tp['f_max']))

            mask[idx_f_min:idx_f_max, idx_t_min:idx_t_max] = 1.0
        return mask

