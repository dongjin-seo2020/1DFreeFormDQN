import gym
import numpy as np
import matplotlib.pyplot as plt
#import matlab.engine
import subprocess


class CustomEnv(gym.Env): 
    
    #initialization
    def __init__(self, n_cells, wavelength, desired_angle):
        super(CustomEnv, self).__init__()
        #self.eng = matlab.engine.start_matlab()
        #self.eng.addpath(self.eng.genpath(r self.path));
        #self.eng.addpath(self.eng.genpath('solvers'));
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.struct = np.ones(self.n_cells)
    
    def getEffofStructure(self, struct, wavelength, desired_angle):


        /usr/local/MATLAB/MATLAB_Runtime/v98 
        effs = subprocess.run(["run_Eval_Eff_1D.sh"],struct, int(wavelength), int(desired_angle))
        return effs 
    
    def step(self, action): #array: input vector, ndarray
        done = False
        #return efficiency and assign it to result_before
        result_before = self.getEffofStructure(self.struct.tolist(), self.wavelength, \
                                          self.desired_angle)
        struct_after= self.struct.copy()
        if action==self.n_cells:
            done=True
        elif (struct_after[action] == 1): #if +1, change to -1, vice versa.
            struct_after[action] = -1
        elif(struct_after[action] == -1):
            struct_after[action] = 1
        else:
            raise ValueError('struct component should be 1 or -1')
        result_after = self.getEffofStructure(struct_after.tolist(), self.wavelength,\
                                         self.desired_angle)
        
        
        #various reward can be set
        reward = (result_after)**3   
        #reward = result_after - result_before
        observation = result_after
             
        self.struct = struct_after.copy()
        return struct_after.squeeze(), observation, reward, done
        
    def reset(self): #initializing the env
        self.struct = np.ones(self.n_cells) 
        eff_init = self.getEffofStructure(self.struct.tolist(), self.wavelength, \
                                          self.desired_angle)
        self.done = False
        return self.struct.squeeze(), eff_init
    
    def get_obs(self):
        return tuple(self.struct) 
        
    def render(self, mode= 'human', close = False):
        plt.plot(self.struct)

        
