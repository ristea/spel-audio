import torch.utils.data
import soundfile as sf
import glob
import os
import numpy as np
from data.base_dataset import MelSpecComputer


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.sounds_path = list(glob.glob(os.path.join(config['test_data_path'], "*.flac")))

        self.window = config['windows_size_fs'] * config['sr']
        self.stride = config['windows_stride'] * config['sr']

        self.mel_spec_computer = MelSpecComputer(sr=config['sr'], fmin=config['f_min'], fmax=config['f_max'],
                                                 n_mels=config['n_mels'], new_sr=config['new_sr'])

    def __getitem__(self, index):
        # Compute data
        raw_signal, fs = sf.read(self.sounds_path[index])
        raw_signal = np.stack([raw_signal[i:i + self.window] for i in range(0, 60 * fs - self.window, self.stride)])

        res_spect = []
        for i in range(0, len(raw_signal)):
            spect = self.mel_spec_computer(raw_signal[i])
            res_spect.append(spect)
        spectrogram = np.array(res_spect)

        return spectrogram, self.sounds_path[index].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.sounds_path)

    def __repr__(self):
        return self.__class__.__name__
