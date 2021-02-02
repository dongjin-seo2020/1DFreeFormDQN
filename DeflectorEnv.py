import gym
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine



class CustomEnv(gym.Env): 
    
    #initialization
    def __init__(self, n_cells, wavelength=900, desired_angle=60):
        super(CustomEnv, self).__init__()
        self.eng = matlab.engine.start_matlab()
        
        self.path = ''
        self.eng.addpath(self.eng.genpath(r self.path));
        self.eng.addpath(self.eng.genpath('solvers'));
        self.n_cells = n_cells
        self.wavelength = matlab.double([wavelength])
        self.desired_angle = matlab.double([desired_angle])
        self.struct = np.ones(self.n_cells)
    
    def getEffofStructure(self, struct, wavelength, desired_angle):
        effs = self.eng.Eval_Eff_1D(struct, wavelength, desired_angle)
        return effs
    
    def step(self, action): #array: input vector, ndarray
        done = False
        #efficiency를 리턴받아 result_before에 할당
        result_before = self.getEffofStructure(matlab.double(self.struct.tolist()), self.wavelength, \
                                          self.desired_angle)
        struct_after= self.struct.copy()
        if action==self.n_cells:
            done=True
        elif (struct_after[action] == 1): #1이면 -1로 만들고 -1이면 1으로 만든다
            struct_after[action] = -1
        elif(struct_after[action] == -1):
            struct_after[action] = 1
        else:
            raise ValueError('struct component should be 1 or -1')
        result_after = self.getEffofStructure(matlab.double(struct_after.tolist()), self.wavelength,\
                                         self.desired_angle)
        #reward = result_after - result_before
        
        reward = (result_after)**3
        
        #reward = 1-(1-result_after)**3
        observation = result_after ##check!
             
        self.struct = struct_after.copy()
        return struct_after.squeeze(), observation, reward, done
        
    def reset(self): #initializing the env
        self.struct = np.ones(self.n_cells) 
        eff_init = self.getEffofStructure(matlab.double(self.struct.tolist()), self.wavelength, \
                                          self.desired_angle)
        self.done = False
        return self.struct.squeeze(), eff_init
    
    def get_obs(self):
        return tuple(self.struct)  #
        
    def render(self, mode= 'human', close = False):
        plt.plot(self.struct)

        
