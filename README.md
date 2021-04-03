# 1DFreeFormDQN
Design code of 1D free-from deflector metasurface

## target condition
multiple angles / broadband wavelength learned by a single network

FC: learns the correlation of inputs and infers the maximum Q action (stabilized by Q network & Target Q network) 

## original code condition
deflection angle 60 degree / wavelength 900nm (from [Fan group sample inference case](https://github.com/jonfanlab/GLOnet))

## simulation
The simulation which corresponds to the environment in RL framework runs on MATLAB RCWA open source [Reticolo](https://zenodo.org/record/3610175#.YBkECS2UGX0)

## example
~~~
python main.py --wavelength=900 --angle=60 --eps_greedy_period=1000000
~~~


## installation
If you install it without any version control of environments, type 
~~~
pip install requirements.txt
~~~

or for Anaconda,
~~~
conda install requirements.txt
~~~
