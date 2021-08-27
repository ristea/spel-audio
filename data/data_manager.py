from data.base_dataset import BaseDataset
import numpy as np
import torch

from data.spl_dataset import SPLDataset
from data.test_dataset import TestDataset


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_test_dataloader(self):
        dataset = TestDataset(self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        return dataloader

    def get_spl_dataloader(self):
        dataset = SPLDataset(self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        return dataloader

    def get_train_eval_dataloaders(self, path, idx=0):
        np.random.seed(777 + idx)

        dataset = BaseDataset(path, self.config)
        dataset_size = len(dataset.sounds_path)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        train_size = int(train_split * dataset_size)

        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        val_indices = indices[train_size:]
        train_indices = indices[:train_size]
        train_indices = train_indices + list(range(len(dataset.sounds_path),
                                                   len(dataset.sounds_path) + len(dataset.sounds_path_spl)))

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler)
        return train_loader, validation_loader
