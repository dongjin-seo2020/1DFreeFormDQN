#borrowed from minimalrl-pytorch written by seungeunrho, et al.
#source: https://github.com/seungeunrho/minimalRL
#little modification was applied by dongjinseo-2020

import collections
import random
import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_limit):
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

        return (torch.tensor(s_lst, dtype=torch.float),
                torch.tensor(a_lst), torch.tensor(r_lst),
                torch.tensor(s_prime_list, dtype=torch.float),
                torch.tensor(done_mask_list))

    def size(self):
        return len(self.buffer)
