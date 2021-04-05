# 1DFreeFormDQN
Design code of 1D free-from deflector metasurface


![plot](./images/schematics.png)

## target condition
multiple angles / wavelengths learned by a single network

The performance was checked for: wavelength of 900nm, 1000nm, 1100nm / deflection angle of 50°, 60°, 70°. Please refer to the paper(link) for further information.

FC: learns the correlation of inputs and infers the maximum Q action (stabilized by Q network & Target Q network) 

## algorithm

The code utilizes Deep Q Network, which is a basic algorithm of Reinforcement Learning.

In addition to final efficiency of structure generated by the algorithm, we also plotted the "validation" efficiency, which is an average value of 10(can vary) episodes with epsilon value 1%. This is for decreasing the stochasity generated by epsilon and observe the development of algorithm as the process goes by.

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

## optimized structures
The optimized structures are saved as .np files in ./structure folder
