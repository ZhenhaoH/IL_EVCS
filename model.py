from torch import nn
import torch


def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.normal_(0.0, 0.01)


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']
        self.hidden3_units = args['hidden3']
        self.e_min = args['e_min']
        self.e_max = args['e_max']
        self.is_train = args['train']
        self.ha1 = args['ha1']
        self.ha2 = args['ha2']
        self.ha3 = args['ha3']

        n = 64
        self.rnn1 = nn.LSTM(1, n)
        self.rnn2 = nn.LSTM(1, n)
        self.rnn3 = nn.LSTM(1, n)
        self.fc1 = nn.Linear(3 * n + 2, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, self.hidden3_units)
        self.fc4 = nn.Linear(self.hidden3_units, self.action_dim)
        self.act_fn = nn.ReLU()
        self.act = nn.Hardtanh(self.e_min, self.e_max)

    def forward(self, x):
        price = x[:, : self.ha1].transpose(0, 1)
        price = price.contiguous().view(self.ha1, -1, 1)
        pv = x[:, self.ha1: self.ha1+self.ha2].transpose(0, 1)
        pv = pv.contiguous().view(self.ha2, -1, 1)
        load = x[:, self.ha1+self.ha2: self.ha1+self.ha2+self.ha3].transpose(0, 1)
        load = load.contiguous().view(self.ha3, -1, 1)
        out1, _ = self.rnn1(price)
        out2, _ = self.rnn2(pv)
        out3, _ = self.rnn3(load)
        x = torch.concat([out1[-1,:,:], out2[-1,:,:], out3[-1,:,:], x[:,-2:]], dim=1)
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.act_fn(self.fc3(x))
        x = self.act(self.fc4(x))
        return x

    def initialize(self):
        self.apply(initWeights)
