import gym
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import S4
import time

class CustomEnv(gym.Env): 
    
    #initialization
    def __init__(self, n_cells, wavelength, desired_angle):
        
        super(CustomEnv, self).__init__()
        self.thick0 = 325
        self.wavelength = 900
        self.angle = 70
        self.nG = 40 
        self.period = abs(self.wavelength/np.sin(self.angle/180*np.pi))
        self.freq = 1/self.wavelength
        self.S = S4.New(Lattice=((self.period,0),(0,0)), NumBasis=self.nG)
        self.n_cells = n_cells
        self.wavelength = int(wavelength)
        self.desired_angle = int(desired_angle)
        self.struct = np.ones(self.n_cells)
        self.Nx = np.size(self.struct)
        self.eff = 0

        self.S.SetMaterial(Name = 'Si', Epsilon = 3.635**2)
        self.S.SetMaterial(Name = 'glass', Epsilon = 1.45**2)
        self.S.SetMaterial(Name = 'air', Epsilon = 1**2)
        self.S.AddLayer(Name = 'glass', Thickness= 0, Material='glass')
        self.S.AddLayer(Name = 'grating', Thickness= self.thick0, Material='air')
        self.S.AddLayer(Name = 'air', Thickness=0, Material='air')

        self.S.SetOptions(
            PolarizationDecomposition = True,
        )
        self.S.SetExcitationPlanewave(
            IncidenceAngles=(
                    0, # polar angle in [0,180)
                    0  # azimuthal angle in [0,360)
            ),
            sAmplitude = 0,
            pAmplitude = 1,
            Order = 0
        )
        self.S.SetFrequency(self.freq)

    def getEffofStructure(self, struct, wavelength, desired_angle):
        
        self.struct = self.struct/2 + 0.5
        for i in range(np.size(self.struct)):
            if self.struct[i]==1:
                self.S.SetRegionRectangle(
                    Layer = 'grating',
                    Material = 'Si',
                    Center = (-self.period/2+self.period/(2*self.Nx) + i*(self.period/self.Nx), 0),
                    Angle = 0,
                    Halfwidths = (self.period/(2*self.Nx), 0)
                    )
        
        
        (fi, bi) = self.S.GetPoyntingFlux(Layer = 'glass')
        (fo, bo) = self.S.GetPoyntingFlux(Layer = 'air')
        
        effs = np.real(fo/fi) 
        return effs
    
    def step(self, action): #array: input vector, ndarray
        done = False
        #efficiency를 리턴받아 result_before에 할당
        result_before = self.eff
        struct_after= self.struct.copy()
        if action==self.n_cells:
            done=True
        elif (struct_after[action] == 1): #1이면 -1로 만들고 -1이면 1으로 만든다
            struct_after[action] = -1
        elif(struct_after[action] == -1):
            struct_after[action] = 1
        else:
            raise ValueError('struct component should be 1 or -1')
        self.eff = self.getEffofStructure(self.struct, self.wavelength,\
                                         self.desired_angle)
        #reward = result_after - result_before
        
        reward = 4*(self.eff-result_before)
        
        #reward = 1-(1-result_after)**3
             
        self.struct = struct_after.copy()
        
        return struct_after.squeeze(), self.eff, reward, done
        
        
    def reset(self): #initializing the env
        self.struct = np.ones(self.n_cells) 
        eff_init = self.getEffofStructure(self.struct, self.wavelength, \
                                          self.desired_angle)
        self.done = False
        return self.struct.squeeze(), eff_init
    
    def get_obs(self):
        return tuple(self.struct) 
        
    def render(self, mode= 'human', close = False):
        plt.plot(self.struct)
