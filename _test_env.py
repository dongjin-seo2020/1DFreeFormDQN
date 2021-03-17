import gym
from DeflectorEnv_Reticolo_Linux import CustomEnv


wavelength = 900
angle = 60
n_cells = 64
env = CustomEnv(n_cells, wavelength, angle)

for i in range(8):
    struct, eff, _ , _ = env.step(i)
    print(eff)
    print(struct)

