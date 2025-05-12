from cmath import inf
import numpy as np
import torch
import argparse
import random
import wandb

from agent import Agent


def train(main_args):
    project, name, epochs, tolerance = main_args.project, main_args.name, main_args.epochs, main_args.tolerance
    save_name = '{}/{}'.format(project, name)
    args = {
        'save_name': save_name,
        'adam': main_args.adam,
        'hidden1': 512,
        'hidden2': 256,
        'hidden3': 128,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'train': True,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')
    
    seed = 0
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    agent = Agent(device, args)
    
    wandb.init(project='Training')
    
    trajectories = []
    traj_len = []
    trajectories_val = []
    traj_len_val = []
    
    # replace with your data
    data_train = np.append(np.load('scenarios/pjm/2019.npy'), np.load('scenarios/pjm/2020.npy'), axis=2)
    data_train = np.append(data_train, np.load('scenarios/pjm/2021.npy'), axis=2)
    price_train = np.append(np.load('scenarios/california iso/2019.8.npy'), np.load('scenarios/california iso/2020.8.npy'), axis=1)[:, : 550]
    load_train = np.load('scenarios/uk power network/load.npy')[:, :550]
    pv_train = data_train[:, 1, 212: 762]
    samp = random.sample(range(0, price_train[0, :].size), 542)
    price_val = price_train[:, samp]
    pv_val = pv_train[:, samp]
    load_val = load_train[:, samp]
    
    for n in range(price_train[0, :].size - 4):
        price = price_train[:, n: n+5].flatten('F')
        pv = pv_train[:, n: n+5].flatten('F')
        load = load_train[:, n: n+5].flatten('F')
        trajectories, traj_len = agent.optimum_generator(trajectories, traj_len, price, pv, load)

    for n in range(price_val[0, :].size - 4):
        price = price_val[:, n: n+5].flatten('F')
        pv = pv_val[:, n: n+5].flatten('F')
        load = load_val[:, n: n+5].flatten('F')
        trajectories_val, traj_len_val = agent.optimum_generator(trajectories_val, traj_len_val, price, pv, load)
    
    best = inf
    for epoch in range(epochs):
        loss, lr = agent.train(trajectories, traj_len)
        loss_val = agent.val(trajectories_val, traj_len_val)
        log_data = {'loss (train)': loss, 'learning rate': lr, 'loss (val)': loss_val}
        print(log_data)
        wandb.log(log_data)
        if loss_val < best:
            agent.save(save_best=True)
            best = loss_val
            best_epoch = epoch
            lowest_loss = loss_val
            early_stop = 0
        else:
            agent.save(save_best=False)
            early_stop += 1
        if early_stop >= tolerance:
            break
    print('The best epoch is ', best_epoch, lowest_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BC')
    parser.add_argument('--project', type=str, default='EVCS', help='save to project/name')
    parser.add_argument('--name', type=str, default='trial', help='save to project/name')
    parser.add_argument('--resume', type=int, default=0, help='resume at # of checkpoint.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs in training')
    parser.add_argument('--adam', action='store_false', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--tolerance', type=int, default=200, help='tolerance for early stopping')
    args = parser.parse_args()
    dict_args = vars(args)
    train(args)