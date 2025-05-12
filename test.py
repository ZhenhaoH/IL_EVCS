import numpy as np
import torch
import argparse
import random
import wandb

from agent import Agent

def test(main_args):
    project, name = main_args.project, main_args.name
    save_name = '{}/{}'.format(project, name)
    args = {
        'save_name': save_name,
        'hidden1': 512,
        'hidden2': 256,
        'hidden3': 128,
        'train': False,
    }
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
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
    
    wandb.init(project='Test')
    
    data_test = np.load('scenarios/pjm/2021.npy')
    price_test = np.load('scenarios/california iso/2020.8.npy')[:, 181:]
    load_test = np.load('scenarios/uk power network/load.npy')[:, 547:]
    pv_test = data_test[:, 1, 28: 212]
    
    costs = np.zeros(6)

    for n in range(price_test[0, :].size - 4):
        trajectory = []
        traj_len = []
        price = price_test[:, n: n+5].flatten('F')
        pv = pv_test[:, n: n+5].flatten('F')
        load = load_test[:, n: n+5].flatten('F')
        trajectory, traj_len = agent.optimum_generator(trajectory, traj_len, price, pv, load)
        loss, cost, cost_gt, cost_baseline, constraint_value, out_of_constraint, loose = agent.test(trajectory)
        costs += np.array([cost, cost_gt, cost_baseline, out_of_constraint, constraint_value, loose], dtype=np.float32)
        
        log_data = {'cost (behaviour cloning)': cost, 'cost (theoretical optimum)': cost_gt, 'cost (baseline)': cost_baseline, 'loss': loss,
                    'cumulative cost (behaviour cloning)': costs[0], 'cumulative cost (theoretical optimum)': costs[1], 'cumulative cost (baseline)': costs[2], 
                    'not full charged': constraint_value, 'out of constraints': costs[3], 'total not full charged': costs[4], 'loose constraints': costs[5]}
        print(log_data)
        wandb.log(log_data)
    print("Cost reduction of BC and TO compared to UC: ", 1 - costs[0] / costs[2], 1 - costs[1] / costs[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BC')
    parser.add_argument('--project', type=str, default='EVCS', help='save to project/name')
    parser.add_argument('--name', type=str, default='trial', help='save to project/name')
    parser.add_argument('--resume', type=int, default=0, help='resume at # of checkpoint.')
    parser.add_argument('--num_traj', type=int, default=0, help='number of trajectories or scenarios')
    parser.add_argument('--adam', action='store_false', help='use torch.optim.Adam() optimizer')
    args = parser.parse_args()
    dict_args = vars(args)
    test(args)
        