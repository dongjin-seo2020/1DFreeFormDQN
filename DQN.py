import gym
from DeflectorEnv_S4 import CustomEnv
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
from collections import namedtuple

import shelve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os


#Tensorboard
import tensorflow as tf


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs') 

################################################################################

SummaryWriterName = '64_900_80'
device_name = os.getcwd() + '/devices/epi{}.png'
np_name = os.getcwd() + '/np_save/epi{}.npy'
saved_fig_name = os.getcwd()+'/summary'
PATH = os.getcwd()+'\\'+'model'+'\\'
q_net_name = PATH+'\\'+'q'
q_target_net_name = PATH+'\\'+'q_target'

################################################################################

#Hyperparameters
learning_rate = 0.001 # 0.0001로 줄이기?
gamma         = 0.99 #0.98
buffer_limit  = 10000000  #### 늘려야함!!!!!!!!!!!!!!!!   ####
batch_size    = 32
n_cells = 64
tau = 1
episode_length =64
EpisodeNumber = 200000
print_interval = 100
n_epi_decreasing_period = 2500
train_start_memory_size = 5000
train_number =1
train_frequency = 2
target_update_frequency = 1000


################################################################################

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_list, done_mask_list = [], [], [], [], []
        
        for transition in mini_batch:   #transition: tuple
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_list, dtype=torch.float), \
                torch.tensor(done_mask_list)
        
    def size(self):
        return len(self.buffer) 
        
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        
        self.fc1 = nn.Linear(n_cells, 2*n_cells)
        #self.fc2 = nn.Linear(2*n_cells, 4*n_cells)
        #self.fc3 =nn.Linear(4*n_cells, 2*n_cells)
        self.fc2 = nn.Linear(2*n_cells, 2*n_cells)
        self.fc3 = nn.Linear(2*n_cells, n_cells+1)  ########################
        self.m = nn.LeakyReLU(0.1)
        
        
        #### leakyReLU -> tanh?
        
    def forward(self, x):
        #print('1: ', x.shape)
        #x = torch.reshape(x, (1,16))
        
        
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = self.fc3(x)
        
        return x
      
    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs,(1,n_cells))
        #print(obs.shape)
        out = self.forward(obs)
        coin = random.random() #0<coin<1
        if coin < epsilon:
            return np.random.randint(0,n_cells+1)
        else:
           # print(out.argmax().item())
            return out.argmax().item()

        
def merge_network_weights(q_target_state_dict, q_state_dict):
    TAU = tau
    dicts = {}
    for k,v in q_target_state_dict.items():
        dicts[k] = q_target_state_dict[k]*(1-TAU)+q_state_dict[k]*(TAU)
    return dicts
   

        
def train(q, q_target, memory, optimizer):
    for i in range(train_number):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)#.to(device)
        q_a = q_out.gather(1,a)#.to(device)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        #print('maxq: ',max_q_prime)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)  #huber loss
        #print('target: ',target.shape, '\nq_a ', q_a,'\nmaxq: ',max_q_prime, '\nloss: ',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
#     env = gym.make('Deflector-v0')
    env = CustomEnv(n_cells)
    q = Qnet()
    q_target = Qnet() 
    q_target.load_state_dict(q.state_dict())
    #writer.add_graph(q)
    #writer.add_graph(q_target)
    memory = ReplayBuffer()

    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    epi_len_st= []
    count =0
    
    nstep =0
    
    for n_epi in range(EpisodeNumber):
        epsilon = max(0.01, 0.9 - 0.1*(n_epi/n_epi_decreasing_period))#
        s, eff_init = env.reset()
        done = False
        eff_epi_st = np.zeros((episode_length,1))
        epi_length = 0
        
        for t in range(episode_length):
            nstep += 1
            q.eval()
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, eff_next, r, done = env.step(a)
            done_mask = 1 - done
            memory.put((s,a,r,s_prime,done_mask))   
            s = s_prime
            eff_epi_st[t] = eff_next
            score += r
            
            writer.add_scalar('episode'+str(episode_length), eff_next, t)
            
            epi_length = t+1
            if done:
                break

            
            if memory.size()>train_start_memory_size and nstep%train_frequency==0 : 
                q.train()
                q_target.train()
                train(q, q_target, memory, optimizer)
            if nstep%target_update_frequency ==0:
                merged_dict = merge_network_weights(q_target.state_dict(),q.state_dict())
                q_target.load_state_dict(merged_dict)

        if n_epi%print_interval==0 and n_epi!=0:
            
            q.score_sum.append(score)
            #q.score_init_final.append(eff_next-eff_init)
            q.effdata.append(eff_next)
            epi_len_st.append(epi_length)
            #############
            print("n_episode :{}, score : {}, eff : {}, effmax : {}, episode length : {}, n_buffer : {}, eps : {:.1f}%".format(
                                                        n_epi, score, eff_next, np.max(eff_epi_st), epi_length, memory.size(), epsilon*100))  #score/print_interval
            #writer.add_scalar('eff', eff_next, n_epi)
            fig_path = device_name.format(n_epi)
            np_path = np_name.format(n_epi)
            plt.imshow(s.reshape(1,-1),cmap='Greys')
            plt.yticks([])
            #plt.colorbar()
            
            #1: purple / #-1: yellow
            #plt.axis('off')
            plt.savefig(fig_path)
            np.save(np_path, s_prime)
        score = 0.0 
            
    
    
    plt.plot(q.effdata)
    plt.title('eff')
    plt.savefig(saved_fig_name+'_eff', format='eps')
    
    plt.plot(epi_len_st)
    plt.title('episode length')
    plt.savefig(saved_fig_name+'_epi_length', format='eps')
    
    plt.plot(s)
    plt.title('final structure')
    plt.savefig(saved_fig_name+'_structure', format='eps')
    
    plt.savefig(saved_fig_name, format='eps')
    ## eps도 plot
    
    print('initial eff: {}'.format(eff_init))
    print('final eff: {}'.format(eff_next))
    print("Qnet's state_dict:")
    for param_tensor in q.state_dict():
        print(param_tensor, "\t", q.state_dict()[param_tensor].size())
    print("Q_target's state_dict:")
    for param_tensor in q_target.state_dict():
        print(param_tensor, "\t", q_target.state_dict()[param_tensor].size())
    print("Qnet's Optimizer's state_dict:")
    for var_name in q.state_dict():
        print(var_name, "\t", q.state_dict()[var_name])
        
    print("Q_target's Optimizer's state_dict:")
    for var_name in q_target.state_dict():
        print(var_name, "\t", q_target.state_dict()[var_name])
        
    torch.save(q.state_dict(), q_net_name)
    torch.save(q_target.state_dict(), q_target_net_name)
    env.close()

if __name__ == '__main__':
    main()
