#from deflector_reticolo import CustomEnv
from deflector_S4 import CustomEnv
import network
from replaybuffer import ReplayBuffer
import logger

import gym
import numpy as np
import argparse
import os
import json
import datetime
import logging
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


if __name__== '__main__':
    
    parser = argparse.ArgumentParser()

    #parameters setup; 
    #also can be done by changing /config/config.json
    parser.add_argument('--nG', default=None, help="diffraction order", type=int)
    parser.add_argument('--wl', default=None, help = "wavelength", type=float)
    parser.add_argument('--ang', default=None, help ="angle", type=float)
    parser.add_argument('--lr', default=None, help="learning rate", type=float)
    parser.add_argument('--gamma', default=None, help='discount factor', type=float)
    parser.add_argument('--buf', default=None, help='buffer size', type=int)
    parser.add_argument('--ncells', default=None, help='number of cells', type=int)
    parser.add_argument('--tau', default=None, help='soft update weight', type=float)
    parser.add_argument('--epilen', default=None, help='episode length', type=int)
    parser.add_argument('--stepnum', default=None, help='overall step number', type=int)
    parser.add_argument('--printint', default=None, help='interval of \
                        printing intermediate outputs', type=int)
    parser.add_argument('--eps_greedy_period', default=None, \
                        help='step number period that decreases epsilon', type=int)

    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--train_start_memory_size', default=None, \
                        help='saved data number in replay buffer that starts\
                            training', type=int)
    parser.add_argument('--train_num', default=None, help='number of \
                        training when train_network() is called', type=int)
    parser.add_argument('--train_step', default=None, help='step period \
                        when train_network() happens', type=int)
    parser.add_argument('--merge_step', default=None, help='step period \
                        when the Q network weights are merged to target network', type=int)                    
    parser.add_argument('--minimum_epsilon', default= None, help='final epsilon value', type=float)
    parser.add_argument('--validation_step', default= None, help='validation step', type=int)
    
    #training or inference
    #parser.add_argument('--train', default=True, help="if True, train. \
    #                    if False, infer only")
    
    #decide wheter to save model weights or not
    parser.add_argument('--save_model', default=True, help='decide wheter to save model weights or not')
                    
    #save model as checkpoint in every episode
    parser.add_argument('--checkpoint', default=False, help="if True, save weights of every episode. \
                        if False, write over same file")

    #load the saved weight
    parser.add_argument('--load_weight', default=False,\
                                            help="weight reload")
    #load config.json
    parser.add_argument('--load_config', default=True, \
                        help = "whether to use config.json")
    
    #decide whether to use tensorboard or not
    parser.add_argument('--tb', default=True, help='tensorboard setting, True/False')

    #decide wheter to save device images or not
    parser.add_argument('--save_devices', default=False, help='decide wheter to save device images or not - jpg')

    #decide wheter to save np structure or not
    parser.add_argument('--save_np_struct', default=False, help='decide wheter to save numpy structure of devices or not')

    
    #decide wheter to save summary or not
    parser.add_argument('--save_summary', default=True, help='decide wheter to save model or not')

    #decide wheter to save source code or not
    parser.add_argument('--source_code_save', default=True, help='decide wheter to save model or not')

    parser.add_argument('--network', default='DQN', help='decide which network to use: "DQN"(default), "Double"(Double DQN), "Dueling"(Dueling DQN)')

    parser.add_argument('--optimizer', default='Adam', help='decide which optimizer to use: Adam, AdamW, SGD, RMSprop')

    parser.add_argument('--save_optimum', default=True, help='decide whether to save the optimal structure or not')

    parser.add_argument('--tag', default='', help='folder tag name for experiments')
    
    args = parser.parse_args()

    path_json = './config/config.json'
    path_devices = '/devices/epi{}.png'
    path_devices_max = '/devices/'
    path_np_struct = '/np_struct/epi{}.npy'
    path_np_struct_max = '/np_struct/'
    path_model = '/model/'
    path_summary = '/summary/'
    path_logs = '/logs/'


    if args.load_config==True:
        
        #bring parameters from json file
        with open(path_json) as f:
            data = json.load(f)  #dict file generated
       
       #assignment of varibles from dict
        for k, v in vars(args).items():
	        if v is None:
		        setattr(args, k, float(data[k]))
        #for k, v in args.__dict__()

	
        print(vars(args))

    t = datetime.datetime.now()
    timeFolderName = t.strftime("%Y_%m_%d_%H_%M_%S")+"/wl"+str(args.wl)+\
            "_angle"+str(args.ang)+"_ncells"+str(int(args.ncells))
    
    filepath = 'experiments/'+args.network+'/'+args.tag+'/'+timeFolderName
    

   
    print('\n File location folder is: %s \n' %filepath)


    os.makedirs(filepath+'/devices', exist_ok=True)
    os.makedirs(filepath+'/np_struct', exist_ok=True)
    os.makedirs(filepath+'/model', exist_ok=True)
    os.makedirs(filepath+'/summary', exist_ok=True)
    os.makedirs(filepath+'/logs', exist_ok=True)
    

    if args.tb==True:
        
        writer = SummaryWriter(filepath+path_logs)
        writer_val = SummaryWriter(filepath+path_logs+'val/')
    ##### setting up the environment
    # Reticolo
    #env = CustomEnv(int(args.ncells), args.wl, args.ang)
    # S4
    env = CustomEnv(int(args.nG),int(args.ncells), args.wl, args.ang)
    

    if args.network=='DQN' or args.network=='Double':
        q = network.Qnet(int(args.ncells))
        q_target = network.Qnet(int(args.ncells))
        if args.load_weight==True:
            weight_loc = input('write down the location of the weight file. ex) "./experiments/DQN/[time]/[condition]/model/~"')
            #TODO
            pass
        q_target.load_state_dict(q.state_dict())

    elif args.network =='Dueling':
        q = network.DuelingQnet(int(args.ncells))
        q_target = network.DuelingQnet(int(args.ncells))
        q_target.load_state_dict(q.state_dict())

    if args.network=='Double' or args.network=='Dueling':
        double=True
    else:
        double=False

    memory = ReplayBuffer(int(args.buf))

    #setting up the optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(q.parameters(), lr=args.lr)

    elif args.optimizer == 'Nadam':
        # import optim
    # optim has much more optimizers than torch.optim
        pass

    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(q.parameters(), lr=args.lr)

    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(q.parameters(), lr=args.lr)

    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(q.parameters(), lr=args.lr)



    epi_len_st= []
    n_epi =0    
    count = 0
    
    #logger handler
    loggername = filepath+path_logs+'log'
    lgr = logging.getLogger(loggername)
    sh = logging.StreamHandler()
    lgr.addHandler(sh)
    lgr.setLevel(logging.DEBUG)
    

    if args.source_code_save == True:
        
        ### TODO code file copy to 'logs' folder
        shutil.copy(os.getcwd()+'/main.py',filepath+path_logs+'main.py')
        shutil.copy(os.getcwd()+'/logger.py',filepath+path_logs+'logger.py')
        shutil.copy(os.getcwd()+'/network.py',filepath+path_logs+'network.py')
        #shutil.copy(os.getcwd()+'/deflector_reticolo.py',filepath+path_logs+'deflector_reticolo.py')
        shutil.copy(os.getcwd()+'/deflector_S4.py',filepath+path_logs+'deflector_S4.py')
        shutil.copy(os.getcwd()+'/replaybuffer.py',filepath+path_logs+'replaybuffer.py')

        
        
    validation_flag = 0
    eff_flag = 0
    while(True):
        s, eff_init = env.reset()
        done = False
        eff_epi_st = np.zeros((int(args.epilen), 1))
        epi_length = 0
        average_reward = 0.0
        
        # episode
        for t in range(int(args.epilen)):
            
            # termination condition
            if count > int(args.stepnum):
                break
            
            if count % args.validation_step == 0 and count != 0:
                validation_flag = 1

            # 
            if validation_flag == 1:

                epsilon = 0.01
                q.eval()
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, eff_next, r, done = env.step(a)
                done_mask = 1 - done
                s = s_prime


                if args.tb==True:
                    writer_val.add_scalar('efficiency / step',
                                    eff_next,
                                    t)
                np.save(filepath+path_np_struct_max+'validation_structure.npy', s)
                np.save(filepath+path_np_struct_max+'validation_efficiency.npy',np.array(eff_next))
                np.save(filepath+path_np_struct_max+'validtion_epi_length.npy', np.array(t))
                        
                
                if t == int(args.epilen):
                    validation_flag = 0

            # when training, make the minimum epsilon as 10% for exploration 
            #if args.train==True:
            else:
                epsilon = max(args.minimum_epsilon, 0.9 * (1. - count / args.eps_greedy_period))
            # when exploting, the epsilon becomes 1%
            #else:
            #    epsilon = 0.01
            
                q.eval()
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, eff_next, r, done = env.step(a)
                done_mask = 1 - done
                memory.put((s, a, r, s_prime, done_mask))
                s = s_prime
                eff_epi_st[t] = eff_next
                average_reward += r

                epi_length = t+1
                count += 1
                
                if args.save_optimum == True: 
                    if eff_next>eff_flag:
                        eff_flag = eff_next
                        plt.figure(figsize=(20,10))
                        plt.imshow(s.reshape(1,-1), cmap = 'Greys')
                        plt.yticks([])
                        plt.savefig(filepath+path_devices_max+'max.png', format = 'png')
                        plt.savefig(filepath+path_devices_max+'max.eps', format = 'eps')
                        np.save(filepath+path_np_struct_max+'max_structure.npy', s)
                        np.save(filepath+path_np_struct_max+'max_efficiency.npy',np.array(eff_flag))
                        np.save(filepath+path_np_struct_max+'max_stepnumber.npy', np.array(count))
                        plt.close()

                if (memory.size() > int(args.train_start_memory_size)
                    and count % int(args.train_step) == 0):
                    
                    q.train()
                    loss = network.train_network(q, q_target, memory, optimizer, int(args.train_num), \
                        int(args.batch_size), args.gamma, double=double)

                    if count % int(args.merge_step) == 0:

                        network.merge_network_weights(q_target.named_parameters(),
                                            q.named_parameters(), args.tau)
                    
            if done:
                break






            if n_epi % int(args.printint) == 0 and n_epi != 0:

                if args.tb==True:
                    writer.add_scalar('one step average reward / episode',
                                    average_reward/epi_length,
                                    n_epi)
                    writer.add_scalar('final step efficiency / episode',
                                    eff_next,
                                    n_epi)
                    writer.add_scalar('episode length / episode',
                                    epi_length,
                                    n_epi)
                    writer.add_scalar('episode length / episode', epi_length, n_epi)
                    writer.add_scalar('epsilon[%] / episode', epsilon*100, n_epi)
                    writer.add_scalar('epsilon[%] / step', epsilon*100, count)
                    writer.add_scalar('efficiency / step', eff_next, count)
                    writer.add_scalar('max efficiency / step', eff_flag, count)
                    writer.add_scalar('memory size / step', memory.size(), count)
                    if (memory.size() > int(args.train_start_memory_size)
                    and count % int(args.train_step) == 0):
                        writer.add_scalar('train loss / step', loss, count)
                q.effdata.append(eff_next)
                epi_len_st.append(epi_length)

                ##TODO: 또 명령어로 들어온 애들 처리
                
                #logging the data: saved in logs+tensorboard folders
                #saved data: hyperparameters(json), logs(csv)
                
                logger.write_logs(loggername, lgr, sh, n_epi, eff_next, \
                    np.max(eff_epi_st), eff_flag, epi_length, memory.size(), epsilon*100, count)
                logger.write_json_hyperparameter(filepath+path_logs, args)

                if args.save_devices == True:
                    logger.deviceplotter(filepath+path_devices, s, n_epi)
                
                if args.save_np_struct == True: 
                    logger.numpystructplotter(filepath+path_np_struct, s, n_epi)

                if args.checkpoint == True:
                    torch.save(q.state_dict(), filepath+path_model+str(count)+'steps_q')
                    torch.save(q_target.state_dict(), filepath+path_model+str(count)+'steps_q_target')
                else:
                    torch.save(q.state_dict(), filepath+path_model+'q')
                    torch.save(q_target.state_dict(), filepath+path_model+'q_target')

        n_epi +=1

        
    if args.save_summary == True:

        logger.summaryplotter(q, epi_len_st, s, filepath+path_summary)
    
    

    

    # TODO : change this part to logger.final_logs()
    if args.save_model == True:
        print('initial eff: {}'.format(eff_init))
        print('final eff: {}'.format(eff_next))
        print("Qnet's state_dict:")
        for param_tensor in q.state_dict():
            print(param_tensor, "\t", q.state_dict()[param_tensor].size())
        print("Q_target's state_dict:")
        for param_tensor in q_target.state_dict():
            print(param_tensor, "\t", q_target.state_dict()[param_tensor].size())
        print("Qnet's Optimizer's state_dict:")
        for var_name in q.state_dict():
            print(var_name, "\t", q.state_dict()[var_name])

        print("Q_target's Optimizer's state_dict:")
        for var_name in q_target.state_dict():
            print(var_name, "\t", q_target.state_dict()[var_name])

        torch.save(q.state_dict(), filepath+path_model+'final_steps_q')
        torch.save(q_target.state_dict(), filepath+path_model+'final_steps_q_target')

        env.close()
	

    print('max efficiency: ', eff_flag)
    print('max stepnumber: ', np.load(filepath+path_np_struct_max+'max_stepnumber.npy'))
    print('max strucutre: ', np.load(filepath+path_np_struct_max+'max_structure.npy'))
