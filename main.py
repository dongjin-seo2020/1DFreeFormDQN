from deflector_reticolo import CustomEnv
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


if __name__== '__main__':
    
    os.makedirs('config', exist_ok=True)
    os.makedirs('devices', exist_ok=True)
    os.makedirs('np_struct', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('summary', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    path_json = './config/config.json'
    path_devices = './devices/epi{}.png'
    path_np_struct = './np_struct/epi{}.npy'
    path_model = './model/'
    path_summary = './summary/'
    path_logs = './logs/'

    parser = argparse.ArgumentParser()

    #parameters setup; 
    #also can be done by changing /config/config.json
    parser.add_argument('--nG', default=None, help="diffraction order")
    parser.add_argument('--wl', default=None, help = "wavelength")
    parser.add_argument('--ang', default=None, help ="angle")
    parser.add_argument('--lr', default=None, help="learning rate")
    parser.add_argument('--gamma', default=None, help='discount factor')
    parser.add_argument('--buf', default=None, help='buffer size')
    parser.add_argument('--ncells', default=None, help='number of cells')
    parser.add_argument('--tau', default=None, help='soft update weight')
    parser.add_argument('--epilen', default=None, help='episode length')
    parser.add_argument('--epinum', default=None, help='episode number')
    parser.add_argument('--printint', default=None, help='interval of \
                        printing intermediate outputs')
    parser.add_argument('--eps_greedy_period', default=None, \
                        help='step number period that decreases epsilon')

    parser.add_argument('--batch_size', default=None)
    parser.add_argument('--train_start_memory_size', default=None, \
                        help='saved data number in replay buffer that starts\
                            training')
    parser.add_argument('--train_num', default=None, help='number of \
                        training when train_network() is called')
    parser.add_argument('--train_step', default=None, help='step period \
                        when train_network() happens')
    parser.add_argument('--merge_step', default=None, help='step period \
                        when the Q network weights are merged to target network')                    
    
    #training or inference
    parser.add_argument('--train', default=True, help="if True, train. \
                        if False, infer only")
    
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

    args = parser.parse_args()

    if args.load_config==True:
        
        #bring parameters from json file
        with open(path_json) as f:
            data = json.load(f)  #dict file generated
       
       #assignment of varibles from dict
        for k, v in vars(args).items():
	        if v is None:
		        setattr(args, k, float(data[k]))
        #for k, v in args.__dict__()

    if args.tb==True:
        t = datetime.datetime.now()
        
        summaryWriterName = t.strftime("%Y_%m_%d_%H_%M_%S")+"_wl"+str(args.wl)+\
            "_angle"+str(args.ang)+"_ncells"+str(int(args.ncells))+args.network
        print('summaryWritername is: %s' %summaryWriterName)
        writer = SummaryWriter(path_logs+summaryWriterName)
    
    ##### setting up the environment
    # Reticolo
    env = CustomEnv(int(args.ncells), args.wl, args.ang)
    # S4
    #env = CustomEnv(int(args.nG),int(args.ncells), args.wl, args.ang)
    

    if args.network=='DQN' or args.network=='Double':
        q = network.Qnet(int(args.ncells))
        q_target = network.Qnet(int(args.ncells))
        if args.load_weight==True:

            #TODO
            pass
        q_target.load_state_dict(q.state_dict())

    elif args.network =='Dueling':
        q = network.DuelingQnet(int(args.ncells))
        q_target = network.DuelingQnet(int(args.ncells))
        q_target.load_state_dict(q.state_dict())

    if args.network=='Double':
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
    
    count = 0
    
    #logger handler
    loggername = path_logs+summaryWriterName+'_logs'
    lgr = logging.getLogger(loggername)
    sh = logging.StreamHandler()
    lgr.addHandler(sh)

    for n_epi in range(int(args.epinum)):
        s, eff_init = env.reset()
        done = False
        eff_epi_st = np.zeros((int(args.epilen), 1))
        epi_length = 0
        average_reward = 0.0
        eff_flag = 0

        for t in range(int(args.epilen)):
            epsilon = max(0.01, 0.9 * (1. - count / int(args.eps_greedy_period)))
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
                    plt.savefig('./devices/max_struct', format = 'png')
                    plt.savefig('./devices/max_struct', format = 'eps')
                    np.save('./np_struct/max_struct.npy',s)
                    np.save('./np_struct/efficiency.npy',np.array(eff_flag))
                    

            if args.train == True:
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
                writer.add_scalar('efficency / step', eff_next, count)
                writer.add_scalar('max efficiency / step', np.max(eff_epi_st), count)
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
                np.max(eff_epi_st), epi_length, memory.size(), epsilon*100, count)
            logger.write_json_hyperparameter(path_logs+summaryWriterName, args)

            if args.save_devices == True:
                logger.deviceplotter(path_devices, s, n_epi)
            
            if args.save_np_struct == True: 
                logger.numpystructplotter(path_np_struct, s, n_epi)

            if args.checkpoint == True:
                torch.save(q.state_dict(), path_model+summaryWriterName+'/'+str(count)+'steps_q')
                torch.save(q_target.state_dict(), path_model+summaryWriterName+'/'+str(count)+'steps_q_target')

    if args.save_summary == True:

        logger.summaryplotter(q, epi_len_st, s, path_summary)
    
    if args.source_code_save == True:
        
        ### TODO os. file copy to 'code' folder

        pass
    

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

        torch.save(q.state_dict(), path_model+summaryWriterName+'/'+'final_steps_q')
        torch.save(q_target.state_dict(), path_model+summaryWriterName+'/'+'final_steps_q_target')

        env.close()
