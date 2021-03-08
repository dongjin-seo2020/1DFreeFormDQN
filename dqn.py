import random
import numpy as np
import torch
import torch.nn as nn



class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.effdata = []
        self.score_sum = []
        self.score_init_final = []

        self.fc1 = nn.Linear(n_cells, 2*n_cells)
        self.fc2 = nn.Linear(2*n_cells, 2*n_cells)
        self.fc3 = nn.Linear(2*n_cells, n_cells+1)
        self.m = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, n_cells))
        #print(obs.shape)
        out = self.forward(obs)
        coin = random.random() #0<coin<1
        if coin < epsilon:
            return np.random.randint(0, n_cells+1)
        else:
           # print(out.argmax().item())
            return out.argmax().item()


def merge_network_weights(q_target_state_dict, q_state_dict, TAU):
    '''
    dicts = {}
    for k,v in q_target_state_dict.items():
        dicts[k] = q_target_state_dict[k] * (1-tau) + q_state_dict[k] * tau
    return dicts
    '''
    dict_dest = dict(q_target_state_dict)
    for name, param in q_state_dict:
        if name in dict_dest:
            dict_dest[name].data.copy_((1 - TAU) * dict_dest[name].data
                                       + TAU * param)


def train(q, q_target, memory, optimizer):
    double = True
    for i in range(train_number):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)#.to(device)
        q_a = q_out.gather(1, a)#.to(device)
        if double:
            max_a_prime = q(s_prime).argmax(1, keepdim=True)
            with torch.no_grad():
                max_q_prime = q_target(s_prime).gather(1, max_a_prime)
        else:
            with torch.no_grad():
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        #print('maxq: ',max_q_prime)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)  #huber loss
        #print('target: ',target.shape, '\nq_a ', q_a,'\nmaxq: ',max_q_prime, '\nloss: ',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


