from math import e
from mip import Model, xsum, minimize, CONTINUOUS, CBC
import numpy as np
import scipy.stats as stats
import torch
import os

from model import Policy

load_min, load_max = 0, 5.5
pv_min, pv_max = 0, 10
price_min, price_max = 0.008, 0.08
e_min, e_max = -7, 7
soc_min, soc_max = 4, 40
charge_efficiency, discharge_efficiency = 0.98, 0.98
c1, c2 = 0.005, -3
E_max = 100
eta_s = 0.6
ha1 = 24
ha2 = 24
ha3 = 24
ha_all = 72

class Agent():
    def __init__(self, device, args):
        
        self.device = device
        self.checkpoint_dir = args['save_name']
        self.is_train = args['train']

        self.state_dim = ha1 + ha2 + ha3 + 2
        self.action_dim = 1
        self.action_bound_min = torch.tensor(e_min, device=device)
        self.action_bound_max = torch.tensor(e_max, device=device)
        args['state_dim'] = self.state_dim
        args['action_dim'] = self.action_dim
        args['e_min'] = e_min
        args['e_max'] = e_max
        args['ha1'] = ha1
        args['ha2'] = ha2
        args['ha3'] = ha3
        
        self.policy = Policy(args).to(device)
        if self.is_train:
            self.lr = args['lr']
            self.weight_decay = args['weight_decay']
            self.adam = args['adam']
            self.momentum = args['momentum']
            
            if self.adam:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, weight_decay = self.weight_decay, betas=(self.momentum, 0.999))
            else:
                self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr, weight_decay = self.weight_decay, momentum=self.momentum)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 1000, 2000, 5000], gamma=0.5)
        self.load()

    def optimum_generator(self, trajectories, traj_len, prices, pv, load):
        self.arrival_time = int(np.around(stats.truncnorm.rvs(-3, 3, loc=18, scale=1, size=1)))
        self.departure_time = int(np.around(stats.truncnorm.rvs(-3, 3, loc=8, scale=1, size=1)))
        soc_t_arr = float(np.around(stats.truncnorm.rvs(-0.6, 0.6, loc=0.5 * soc_max, scale=0.5 * soc_max, size=1), decimals=2))
        self.all_prices = prices
        self.all_pv = pv
        self.all_load = load
        P_b_t = self.all_prices[self.arrival_time + ha_all: self.departure_time + ha_all + 25]
        P_s_t = eta_s * P_b_t
        pv_t = self.all_pv[self.arrival_time + ha_all: self.departure_time + ha_all + 25]
        load_t = self.all_load[self.arrival_time + ha_all: self.departure_time + ha_all + 25]
        slot = self.departure_time - self.arrival_time + 24
        omega = c1 * (e**(c2*(np.arange(slot)+1)/slot)-1)/(e**c2-1)
        
        m = Model(solver_name=CBC)

        a_ch_t = [m.add_var(lb=0, ub=e_max, var_type=CONTINUOUS) for _ in range(slot)]
        a_dis_t = [m.add_var(lb=e_min, ub=0, var_type=CONTINUOUS) for _ in range(slot)]
        E_b_t = [m.add_var(lb=0, var_type=CONTINUOUS) for _ in range(slot)]
        E_s_t = [m.add_var(lb=0, var_type=CONTINUOUS) for _ in range(slot)]
        soc_t = [m.add_var(lb=soc_min, ub=soc_max, var_type=CONTINUOUS) for _ in range(slot)]

        # m.objective = minimize(xsum(P_b_t[i] * E_b_t[i] - P_s_t[i] * E_s_t[i] + omega[i]*(soc_max-soc_t[i]) for i in range(slot)))
        m.objective = minimize(xsum(P_b_t[i] * E_b_t[i] - P_s_t[i] * E_s_t[i] for i in range(slot)))
        for i in range(slot):
            m += E_b_t[i] + pv_t[i] == E_s_t[i] + a_ch_t[i] + a_dis_t[i] + load_t[i]
            
            if i == 0:
                m += soc_t[i] == soc_t_arr
            else:
                m += soc_t[i] == soc_t[i - 1] + charge_efficiency * a_ch_t[i - 1] + a_dis_t[i - 1] / discharge_efficiency
                
            m += soc_t[i] - soc_max <= 0
            m += soc_t[i] - soc_min >= 0
            m += E_b_t[i] + E_s_t[i] <= E_max
            if i == slot - 1:
                m += soc_t[i] + charge_efficiency * a_ch_t[i] + a_dis_t[i] / discharge_efficiency - soc_max == 0

        m.optimize()
        
        i = 0
        j = 0
        result = np.zeros((5, slot))
        result_f = np.zeros((4, slot))
        
        for v in m.vars:
            result[i][j] = v.x
            j += 1
            if j == slot:
                i += 1
                j = 0

        for i in range(1, 5):
            for j in range(slot):
                if i == 1:
                    result_f[i - 1][j] = result[i][j] if result[i][j] != 0 else result[i - 1][j]
                else:
                    result_f[i - 1][j] = result[i][j]
        
        # get all states and actions including the departure time
        action = result_f[0, :].reshape((-1))
        soc = result_f[3, :].reshape((-1))
        for i in range(soc.size):
            t = self.arrival_time + i
            if t < 24:
                a = np.array([t, soc[i]])
            else:
                a = np.array([t-24, soc[i]])
            s = np.append(self.all_pv[t + ha_all + 1 - ha2: t + ha_all + 1], self.all_load[t + ha_all + 1 - ha3: t + ha_all + 1])
            s = np.append(s, a)
            state = np.append(self.all_prices[t + ha_all + 1 - ha1: t + ha_all + 1], s)
            trajectories.append([state, np.array([action[i]])])
        traj_len.append(soc.size)
        
        return trajectories, traj_len
    
    
    
    def train(self, trajectories, traj_len):
        lr = self.scheduler.get_last_lr()[0]
        
        # extract states and actions (ground truth)
        gt_states = np.array([traj[0] for traj in trajectories])
        gt_actions = np.array([traj[1] for traj in trajectories])
        traj_len = np.array([traj_l for traj_l in traj_len])
        # convert to tensor
        gt_states_tensor = torch.tensor(gt_states, device=self.device, dtype=torch.float)
        gt_actions_tensor = torch.tensor(gt_actions, device=self.device, dtype=torch.float)
        
        price_norm = (gt_states_tensor[:, : -(ha2+ha3+2)] - price_min) / (price_max - price_min)
        pv_norm = (gt_states_tensor[:, -(ha2+ha3+2): -(ha3+2)] - pv_min) / (pv_max - pv_min)
        load_norm = (gt_states_tensor[:, -(ha3+2): -2] - load_min) / (load_max - load_min)
        t_norm = gt_states_tensor[:, -2] / 24
        soc_norm = (gt_states_tensor[:, -1] - soc_min) / (soc_max - soc_min)
        states_tensor = torch.cat((price_norm, pv_norm), 1)
        states_tensor = torch.cat((states_tensor, load_norm), 1)
        states_tensor = torch.cat((states_tensor, t_norm.view(-1, 1)), 1)
        states_tensor = torch.cat((states_tensor, soc_norm.view(-1, 1)), 1)
        
        actions_tensor = self.policy(states_tensor)
        loss = torch.sqrt(torch.mean(torch.square(actions_tensor - gt_actions_tensor)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        np_loss = loss.detach().cpu().numpy()
        loss = float(np_loss)
        
        return loss, lr
        
        
    def val(self, trajectories, traj_len):
        # extract states and actions (ground truth)
        gt_states = np.array([traj[0] for traj in trajectories])
        gt_actions = np.array([traj[1] for traj in trajectories])
        traj_len = np.array([traj_l for traj_l in traj_len])
        
        # convert to tensor
        gt_states_tensor = torch.tensor(gt_states, device=self.device, dtype=torch.float)
        gt_actions_tensor = torch.tensor(gt_actions, device=self.device, dtype=torch.float)
        
        price_norm = (gt_states_tensor[:, : -(ha2+ha3+2)] - price_min) / (price_max - price_min)
        pv_norm = (gt_states_tensor[:, -(ha2+ha3+2): -(ha3+2)] - pv_min) / (pv_max - pv_min)
        load_norm = (gt_states_tensor[:, -(ha3+2): -2] - load_min) / (load_max - load_min)
        t_norm = gt_states_tensor[:, -2] / 24
        soc_norm = (gt_states_tensor[:, -1] - soc_min) / (soc_max - soc_min)
        states_tensor = torch.cat((price_norm, pv_norm), 1)
        states_tensor = torch.cat((states_tensor, load_norm), 1)
        states_tensor = torch.cat((states_tensor, t_norm.view(-1, 1)), 1)
        states_tensor = torch.cat((states_tensor, soc_norm.view(-1, 1)), 1)
        actions_tensor = self.policy(states_tensor)
        
        loss = torch.sqrt(torch.mean(torch.square(actions_tensor - gt_actions_tensor)))
        np_loss = loss.detach().cpu().numpy()
        loss = float(np_loss)
        
        return loss
    
        
    def test(self, trajectory):
        # extract states and actions (ground truth)
        gt_states = np.array([traj[0] for traj in trajectory])
        gt_actions = np.array([traj[1] for traj in trajectory])
        
        # convert to tensor
        gt_states_tensor = torch.tensor(gt_states, device=self.device, dtype=torch.float)
        states_tensor = gt_states_tensor.clone().detach()
        gt_actions_tensor = torch.tensor(gt_actions, device=self.device, dtype=torch.float)
        actions_tensor = gt_actions_tensor.clone().detach()
        
        n = len(trajectory)
        out_of_constraint = torch.tensor([0], device=self.device, dtype=torch.float)
        for i in range(n):
            price_norm = (states_tensor[i, : -(ha2+ha3+2)] - price_min) / (price_max - price_min)
            pv_norm = (states_tensor[i, -(ha2+ha3+2): -(ha3+2)] - pv_min) / (pv_max - pv_min)
            load_norm = (states_tensor[i, -(ha3+2): -2] - load_min) / (load_max - load_min)
            t_norm = states_tensor[i, -2] / 24
            soc_norm = (states_tensor[i, -1] - soc_min) / (soc_max - soc_min)
            states_norm_tensor = torch.cat((price_norm.view(1, -1), pv_norm.view(1, -1)), 1)
            states_norm_tensor = torch.cat((states_norm_tensor.view(1, -1), load_norm.view(1, -1)), 1)
            states_norm_tensor = torch.cat((states_norm_tensor.view(1, -1), t_norm.view(1, 1)), 1)
            states_norm_tensor = torch.cat((states_norm_tensor.view(1, -1), soc_norm.view(1, 1)), 1)
            actions_tensor[i, :] = self.policy(states_norm_tensor.view(1, -1))
            
            # postprocessing  
            if actions_tensor[i, :] >= 0:
                overcharge = actions_tensor[i, :] - (soc_max - states_tensor[i, -1]) / charge_efficiency
                if overcharge > 0:
                    out_of_constraint += overcharge * charge_efficiency
                    actions_tensor[i, :] = (soc_max - states_tensor[i, -1]) / charge_efficiency
                if i == n - 1:
                    soc_t_dep = states_tensor[i, -1] + charge_efficiency * actions_tensor[i, :]
                else:
                    states_tensor[i + 1, -1] = states_tensor[i, -1] + charge_efficiency * actions_tensor[i, :]
            else:
                overdischarge = (soc_min - states_tensor[i, -1]) * discharge_efficiency - actions_tensor[i, :]
                if overdischarge > 0:
                    out_of_constraint += overdischarge / discharge_efficiency
                    actions_tensor[i, :] = (soc_min - states_tensor[i, -1]) * discharge_efficiency
                if i == n - 1:
                    soc_t_dep = states_tensor[i, -1] + actions_tensor[i, :] / discharge_efficiency
                else:
                    states_tensor[i + 1, -1] = states_tensor[i, -1] + actions_tensor[i, :] / discharge_efficiency
            
            if i == n - 1:
                if (soc_max - soc_t_dep) / charge_efficiency >= 0:
                    actions_tensor[i, :] = (soc_max - states_tensor[i, -1]) / charge_efficiency
            else:
                if (soc_max - states_tensor[i + 1, -1]) / charge_efficiency >= e_max * (n - i - 1):
                    actions_tensor[i, :] = (soc_max - states_tensor[i, -1]) / charge_efficiency - e_max * (n - i - 1)
            
            E_net = actions_tensor[i, :] + states_tensor[i, -3] - states_tensor[i, -(ha3+3)]
            if E_net > E_max:
                actions_tensor[i, :] = E_max + states_tensor[i, -(ha3+3)] - states_tensor[i, -3]
            elif E_net < -E_max:
                actions_tensor[i, :] = -E_max + states_tensor[i, -(ha3+3)] - states_tensor[i, -3]
                
                
            if actions_tensor[i, :] >= 0:
                if i == n - 1:
                    soc_t_dep = states_tensor[i, -1] + charge_efficiency * actions_tensor[i, :]
                else:
                    states_tensor[i + 1, -1] = states_tensor[i, -1] + charge_efficiency * actions_tensor[i, :]
            else:
                if i == n - 1:
                    soc_t_dep = states_tensor[i, -1] + actions_tensor[i, :] / discharge_efficiency
                else:
                    states_tensor[i + 1, -1] = states_tensor[i, -1] + actions_tensor[i, :] / discharge_efficiency
        
        loss = torch.sqrt(torch.mean(torch.square(actions_tensor - gt_actions_tensor)))
        
        states = states_tensor.detach().cpu().numpy()
        actions = actions_tensor.detach().cpu().numpy()
        soc_t_dep = soc_t_dep.detach().cpu().numpy()
        out_of_constraint = out_of_constraint.detach().cpu().numpy()
        
        y = states[:, -(ha2+ha3+3)].reshape((-1))         # real-time price
        z = states[:, -1].reshape((-1))
        w = actions.reshape((-1))
        
        cost_baseline = 0
        soc = z[0]
        for i in range(z.size):
            E_net_baseline = (e_max - gt_states[i, -(ha3+3)] + gt_states[i, -3])
            if E_net_baseline > E_max:
                action = E_max + gt_states[i, -(ha3+3)] - gt_states[i, -3]
            else:
                action = e_max
            if soc_max - soc >= action * charge_efficiency:
                E_t = action - gt_states[i, -(ha3+3)] + gt_states[i, -3]
                one_baseline = E_t >= 0
                cost_baseline += y[i] * E_t * one_baseline + eta_s * y[i] * E_t * (1 - one_baseline)
                soc += action * charge_efficiency
            else:
                E_t = (soc_max - soc) / charge_efficiency - gt_states[i, -(ha3+3)] + gt_states[i, -3]
                one_baseline = E_t >= 0
                cost_baseline += y[i] * E_t * one_baseline + eta_s * y[i] * E_t * (1 - one_baseline)
                soc = soc_max
        
        one = (w - gt_states[:, -(ha3+3)] + gt_states[:, -3]) >= 0
        cost = np.sum(y * (w - gt_states[:, -(ha3+3)] + gt_states[:, -3]) * one + eta_s * y * (w - gt_states[:, -(ha3+3)] + gt_states[:, -3]) * (1 - one))
        one_gt = (gt_actions.reshape((-1)) - gt_states[:, -(ha3+3)] + gt_states[:, -3]) >= 0
        cost_gt = np.sum(gt_states[:, -(ha2+ha3+3)].reshape((-1)) * (gt_actions.reshape((-1)) - gt_states[:, -(ha3+3)] + gt_states[:, -3]) * one_gt
                         + eta_s * gt_states[:, -(ha2+ha3+3)].reshape((-1)) * (gt_actions.reshape((-1)) - gt_states[:, -(ha3+3)] + gt_states[:, -3]) * (1 - one_gt))
        
        constraint_value = float(soc_max - soc_t_dep)
        out_of_constraint = float(out_of_constraint) + constraint_value
        out_of_loose_constraint = max([out_of_constraint - 0.1, 0])
        np_loss = loss.detach().cpu().numpy()
        loss = float(np_loss)
        
        return loss, cost, cost_gt, cost_baseline, constraint_value, out_of_constraint, out_of_loose_constraint

        
    def save(self, save_best):
        if save_best:
            torch.save({
                'policy': self.policy.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, self.checkpoint_dir + '/best.pt')
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.checkpoint_dir + '/last.pt')
        print('[save] success.')
        
        
    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = self.checkpoint_dir + '/best.pt'
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            if self.is_train:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.policy.load_state_dict(checkpoint['policy'])
            print('[load] success.')
        else:
            self.policy.initialize()
            print('[load] fail.')
        
        