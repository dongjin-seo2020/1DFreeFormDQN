import gym
from DeflectorEnv import CustomEnv
import matplotlib.pyplot as plt



episode_num = 10
cell_num = 64

env = CustomEnv(cell_num)
    
s_before = env.reset()[0]
env.struct = np.ones(cell_num).squeeze()

eff_after = []
eff_before = []
eff = 0

for j in range(episode_num):
  for i in range(cell_num):
      eff_before.append(eff)
      struct_next, eff, _, _ = env.step(np.random.choice(cell_num,1)[0])
      eff_after.append(eff)
  plt.plot(eff_before)
  plt.savefig(os.getcwd()+'/randomwalkcurve'+str(j)+'.eps', format='eps')
