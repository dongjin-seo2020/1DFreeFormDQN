import gym
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from pathlib import Path
import _pickle as json
import os

class CustomEnv(gym.Env):

    #initialization
    def __init__(self, n_cells, wavelength, desired_angle):
        super(CustomEnv, self).__init__()
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(r'C:\Users\Dongjin\RETICOLO V8'));
        self.eng.addpath(self.eng.genpath('solvers'));
        os.makedirs('data',exist_ok=True)
        self.eff_file_path = 'data/eff_table.json'
        if Path(self.eff_file_path).exists():
            with open(self.eff_file_path, 'rb') as f:
                self.eff_table = json.load(f)
        else:
            self.eff_table = {}
        self.n_cells = n_cells
        self.wavelength = matlab.double([wavelength])
        self.desired_angle = matlab.double([desired_angle])
        self.struct = np.ones(self.n_cells)
        self.eff =0

    def getEffofStructure(self, struct, wavelength, desired_angle):
        effs = self.eng.Eval_Eff_1D(struct, wavelength, desired_angle)
        return effs

    def step(self, action): #array: input vector, ndarray
        done = False
        #efficiency를 리턴받아 result_before에 할당
        result_before = self.eff
        struct_after= self.struct.copy()
        if action==self.n_cells:
            done=True
        if (struct_after[action] == 1): #1이면 -1로 만들고 -1이면 1으로 만든다
            struct_after[action] = -1
        elif(struct_after[action] == -1):
            struct_after[action] = 1
        else:
            raise ValueError('struct component should be 1 or -1')
        key = tuple(struct_after.tolist())
       # print(key)
        if key in self.eff_table:
            self.eff = self.eff_table[key]
        else:
            self.eff = self.getEffofStructure(matlab.double(struct_after.tolist()), self.wavelength,\
                                             self.desired_angle)
            self.eff_table[key] = self.eff
       
        reward = (self.eff)**3
        #various reward can be set
        #reward = (result_after)**3.
        #reward = result_after - result_before
        #reward = 1-(1-result_after)**3

        self.struct = struct_after.copy()

        return struct_after.squeeze(), self.eff, reward, done

    def reset(self): #initializing the env
        self.struct = np.ones(self.n_cells)
        eff_init = 0
        self.done = False
        if self.eff_table:
	        with open(self.eff_file_path, 'wb') as f:
	            json.dump(self.eff_table, f)
        return self.struct.squeeze(), eff_init

    def get_obs(self):
        return tuple(self.struct)

    def render(self, mode= 'human', close = False):
        plt.plot(self.struct)
