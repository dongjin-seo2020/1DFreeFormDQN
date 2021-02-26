import gym
from DeflectorEnv_Reticolo_Linux import CustomEnv

import numpy as np
import os
import subprocess

n_cells=64
struct = str(np.ones(n_cells))
print(struct)
wavelength = str(900)
desired_angle=str(60)
subprocess.Popen('chmod +x ' + os.getcwd()+ '/solvers/Eval_Eff_1D/Eval_Eff_1D/for_redistribution_files_only/Eval_Eff_1D', shell=True)
subprocess.Popen('chmod +x ' + os.getcwd()+ '/solvers/Eval_Eff_1D/Eval_Eff_1D/for_redistribution_files_only/run_Eval_Eff_1D.sh', shell=True)
# effs = subprocess.Popen("./solvers/Eval_Eff_1D/Eval_Eff_1D/for_redistribution_files_only/run_Eval_Eff_1D.sh /usr/local/MATLAB/MATLAB_Runtime/v98 "+struct+' '+wavelength+' '+desired_angle, shell=True)
effs = subprocess.call(["./solvers/Eval_Eff_1D/Eval_Eff_1D/for_redistribution_files_only/run_Eval_Eff_1D.sh", "/usr/local/MATLAB/MATLAB_Runtime/v98", struct, wavelength, desired_angle])

print(effs)

# subprocess.run('chmod +x ./solvers/Eval_Eff_1D/Eval_Eff_1D/for_testing/run_Eval_Eff_1D.sh')
#effs = subprocess.run(["./solvers/Eval_Eff_1D/Eval_Eff_1D/for_testing/run_Eval_Eff_1D.sh"],struct, wavelength, desired_angle)
