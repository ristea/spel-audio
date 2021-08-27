import json
import os

import torch
import torch.optim as optim
from networks.ResNeST import resnest50

import utils.losses as loss_functions
from data.data_manager import DataManager
from trainer import Trainer
from utils.data_logs import save_logs_about


def load_net(checkpoint_path):
    net = resnest50(pretrained=True)
    n_features = net.fc.in_features
    net.fc = torch.nn.Linear(n_features, 24)
    net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net = net.eval()
    return net


def generate_spl_data():
    '''
        This method generates the most confident data, based on model accuracy.
        Afterwards, generated data is added into the training process, as described in paper.
    '''

    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train_size'] = 1
    config['valid_size'] = 0
    config['augment'] = False

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    models_paths = [
        '.../latest_checkpoint_0.pkl'
        '.../latest_checkpoint_1.pkl'
        '.../latest_checkpoint_2.pkl'
    ]

    nets = []
    for i in range(0, len(models_paths)):
        checkpoint = torch.load(os.path.join(models_paths[i]), map_location=config['device'])

        model = resnest50(pretrained=True)
        n_features = model.fc.in_features
        model.fc = torch.nn.Linear(n_features, 24)

        model.load_state_dict(checkpoint['model_weights'])
        model.to(config['device'])
        model.eval()
        nets.append(model)

    trainer = Trainer(None, None, None, None, None, None, config)
    trainer.generate_spl_data(nets, save_data_path=config['spel_data_path'])


def train_single():
    '''
        This method trains a single model. Conventional training.

        If you want to train with SPEL generated data, you should modify the config.json file
        as follows: "add_spl_data" -> true
    '''

    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['k_fold'] = 0

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    model = resnest50(pretrained=True)
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, 24)
    model.to(config['device'])

    criterion = getattr(loss_functions, config['loss_function'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'], last_epoch=-1)

    data_manager = DataManager(config)
    train_loader, validation_loader = data_manager.get_train_eval_dataloaders(config['train_data_path'])

    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)
    trainer.train()
    trainer.test_net()


def train_ensamble():
    '''
        This method trains an ensemble model. Conventional training.

        If you want to train with SPEL generated data, you should modify the config.json file
        as follows: "add_spl_data" -> true
    '''

    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    k_fold_split = 5
    for i in range(0, k_fold_split):
        config['k_fold'] = i

        model = resnest50(pretrained=True)
        n_features = model.fc.in_features
        model.fc = torch.nn.Linear(n_features, 24)
        model.to(config['device'])

        criterion = getattr(loss_functions, config['loss_function'])
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'], last_epoch=-1)

        data_manager = DataManager(config)
        train_loader, validation_loader = data_manager.get_train_eval_dataloaders(config['train_data_path'], idx=i)

        trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)
        trainer.train()

    trainer.test_net_ensemble(k_fold_split)


if __name__ == "__main__":
    train_ensamble()
