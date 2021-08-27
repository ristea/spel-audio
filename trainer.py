import torch
import numpy as np

from data.data_manager import DataManager
from utils.stats_manager import StatsManager
from utils.data_logs import save_logs_train, save_logs_eval
import pandas as pd
import os
import pickle
from copy import deepcopy


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, config):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.best_metric = 0.0
        self.best_loss = 100.0

    def train_epoch(self, epoch):
        running_loss = []
        self.network.train()
        self.config['augment'] = True
        for idx, (inputs, labels, mask) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).float()
            # mask = mask.to(self.config['device']).float()
            # boxes = boxes.to(self.config['device']).float()

            predictions = self.network(inputs)
            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')

                running_loss = []

    def eval_net(self, epoch):
        stats_labels = []
        stats_predictions = []

        running_eval_loss = 0.0
        self.network.eval()
        self.config['augment'] = False
        for idx, (inputs, labels, mask) in enumerate(self.eval_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).float()
            # mask = mask.to(self.config['device']).float()
            # boxes = boxes.to(self.config['device']).float()

            with torch.no_grad():
                predictions = self.network(inputs)

            eval_loss = self.criterion(predictions, labels)
            running_eval_loss += eval_loss.item()

            stats_predictions.append(predictions.detach().cpu().numpy())
            stats_labels.append(labels.detach().cpu().numpy())

        performance = self.stats_manager.get_stats(predictions=stats_predictions, labels=stats_labels)
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        print(f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, Performance = {performance}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'Evaluation loss on epoch {epoch} = {running_eval_loss}, Performance = {performance}')

        if self.best_metric < performance:
            self.best_metric = performance
            self.save_net_state(None, best='performance')

    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'],
                                                 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'])
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=None):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'],
                                        f'latest_checkpoint_{self.config["k_fold"]}.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is not None:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'],
                                        f'best_{best}_{self.config["k_fold"]}.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'],
                                        f'model_epoch_{epoch}_{self.config["k_fold"]}.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)

    @torch.no_grad()
    def test_net(self):
        predictions_stats = []
        names_stats = []

        data_manager = DataManager(self.config)
        test_dataloader = data_manager.get_test_dataloader()

        checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'latest_checkpoint_0.pkl'),
                                map_location=self.config['device'])

        network = self.network
        network.load_state_dict(checkpoint['model_weights'])
        network.eval()
        for idx, (inputs, names) in enumerate(test_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            names_stats = names_stats + list(names)

            for bs_idx in range(0, len(inputs)):
                predictions = network(inputs[bs_idx])
                predictions = predictions.max(0).values
                predictions = predictions.detach().cpu().numpy()

                predictions_stats.append(predictions)

        predictions_stats = np.vstack(predictions_stats)
        submission = pd.read_csv('./data/submission.csv')

        for i in range(0, len(submission)):
            recording_name = submission['recording_id'][i]
            processed_recording_idx = names_stats.index(recording_name)

            submission.loc[i, 's0':'s23'] = predictions_stats[processed_recording_idx]

        submission.to_csv(os.path.join(self.config['exp_path'], self.config['exp_name'], 'new_submission.csv'),
                          index=False)

    @torch.no_grad()
    def test_net_ensemble(self, k_fold_split):
        predictions_stats = []
        names_stats = []

        data_manager = DataManager(self.config)
        test_dataloader = data_manager.get_test_dataloader()

        nets = []
        for i in range(0, k_fold_split):
            checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'],
                                                 f'latest_checkpoint_{i}.pkl'),
                                    map_location=self.config['device'])

            network = deepcopy(self.network)
            network.load_state_dict(checkpoint['model_weights'])
            network.eval()
            nets.append(network)

        for idx, (inputs, names) in enumerate(test_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            names_stats = names_stats + list(names)

            for bs_idx in range(0, len(inputs)):
                predictions_ens = 0.
                for net in nets:
                    predictions = net(inputs[bs_idx])
                    predictions = predictions.max(0).values
                    predictions = predictions.detach().cpu().numpy()
                    predictions_ens += predictions

                predictions_ens /= len(nets)
                predictions_stats.append(predictions_ens)

        predictions_stats = np.vstack(predictions_stats)
        submission = pd.read_csv('./data/submission.csv')

        for i in range(0, len(submission)):
            recording_name = submission['recording_id'][i]
            processed_recording_idx = names_stats.index(recording_name)

            submission.loc[i, 's0':'s23'] = predictions_stats[processed_recording_idx]

        submission.to_csv(os.path.join(self.config['exp_path'], self.config['exp_name'], 'new_submission.csv'),
                          index=False)

    @torch.no_grad()
    def generate_spl_data(self, nets, save_data_path='./data/'):
        data_manager = DataManager(self.config)
        spl_dataloader = data_manager.get_spl_dataloader()

        for idx, inputs in enumerate(spl_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()

            for bs_idx in range(0, len(inputs)):

                for win_idx in range(0, len(inputs[bs_idx])):
                    predictions_ens = 0.
                    for net in nets:
                        predictions = net(inputs[bs_idx][win_idx].unsqueeze(0))
                        predictions = predictions.detach().cpu().numpy()
                        predictions_ens += predictions

                    predictions_ens /= len(nets)
                    predictions_ens = predictions_ens[0]
                    predictions_ens[predictions_ens < self.config['spl_th']] = 0
                    if len(predictions_ens[predictions_ens > self.config['spl_th']]):
                        spl_data = {
                            "data": inputs[bs_idx][win_idx].detach().cpu().numpy(),
                            "label": predictions_ens
                        }

                        pickle.dump(spl_data, open(os.path.join(save_data_path, f"{idx}_{bs_idx}_{win_idx}.pkl"), "wb"))
