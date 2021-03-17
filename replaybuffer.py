#borrowed from minimalrl-pytorch written by seungeunrho, et al.
#source: https://github.com/seungeunrho/minimalRL
#little modification was applied by dongjinseo-2020

import collections
import random
import torch
import numpy as np

class ReplayBuffer_deque():
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

#written by namdw
class ReplayBuffer():
    def __init__(self, buffer_limit, n_cells=64):
        self.s = np.ones((buffer_limit, n_cells), dtype=np.uint8)
        self.a = np.ones((buffer_limit, 1), dtype=np.uint8)
        self.r = np.ones((buffer_limit, 1), dtype=np.uint8)
        self.d = np.ones((buffer_limit, 1), dtype=np.uint8)
        self.s_ = np.ones((buffer_limit, n_cells), dtype=np.uint8)
        
        self.max_size = buffer_limit
        self.curr_idx = 0
        self.curr_size = 0

    def put(self, transition):
        s, a, r, d, s_ = transition
        self.s[self.curr_idx] = s
        self.a[self.curr_idx] = a
        self.r[self.curr_idx] = r
        self.d[self.curr_idx] = d
        self.s_[self.curr_idx] = s_
        self.curr_size = max(self.curr_size + 1, self.max_size)
        self.curr_idx = (self.curr_idx + 1) % self.max_size

    def sample(self, n):
        random_idx = np.random.randint(self.curr_size, size=n)
        s = torch.from_numpy(self.s[random_idx]).float()
        a = torch.from_numpy(self.a[random_idx]).long()
        r = torch.from_numpy(self.r[random_idx]).float()
        d = torch.from_numpy(self.d[random_idx]).float()
        s_ = torch.from_numpy(self.s_[random_idx]).float()
        
        return (s, a, r, d, s_)

    def size(self):
        return self.curr_size
