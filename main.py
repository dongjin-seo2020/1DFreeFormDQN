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

    #decide wheter to save model weights or not
    parser.add_argument('--save_model', default=True, help='decide wheter to save model weights or not')

    #decide wheter to save summary or not
    parser.add_argument('--save_summary', default=True, help='decide wheter to save model or not')

    #decide wheter to save source code or not
    parser.add_argument('--source_code_summary', default=True, help='decide wheter to save model or not')

    parser.add_argument('--network', default='DQN', help='decide which network to use: "DQN"(default), "Double"(Double DQN), "Dueling"(Dueling DQN)')

    parser.add_argument('--optimizer', default='Adam', help='decide which optimizer to use')

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
            "_angle"+str(args.ang)+"_ncells"+str(int(args.ncells))
        print('summaryWritername is: %s' %summaryWriterName)
        writer = SummaryWriter(path_logs+summaryWriterName)
    
    #setting up the environment
    env = CustomEnv(int(args.nG), int(args.ncells), args.wl, args.ang)

    if args.network=='DQN' or network=='Double':
        q = network.Qnet(int(args.ncells))
        q_target = network.Qnet(int(args.ncells))
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

    epi_len_st= []
    
    average_reward = 0.0
    count = 0
    for n_epi in range(int(args.epinum)):
        s, eff_init = env.reset()
        done = False
        eff_epi_st = np.zeros((int(args.epilen), 1))
        epi_length = 0
        
        for t in range(int(args.epilen)):
            epsilon = max(0.01, 0.9 * (1. - count / args.eps_greedy_period))
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

            if (memory.size() > args.train_start_memory_size
                and count % args.train_step == 0):

                q.train()
                network.train_network(q, q_target, memory, optimizer, args.train_num, \
                    args.batch_size, args.gamma, double=double)

            if count % args.merge_step == 0:

                network.merge_network_weights(q_target.named_parameters(),
                                      q.named_parameters(), args.tau)
            
            if done:
                break

        if n_epi % int(args.printint) == 0 and n_epi != 0:

            if args.tb==True:
                writer.add_scalar('average reward',
                                average_reward,
                                n_epi)
                writer.add_scalar('final step efficiency',
                                eff_next,
                                n_epi)
                writer.add_scalar('episode length',
                                epi_length,
                                n_epi)
                

            q.effdata.append(eff_next)
            epi_len_st.append(epi_length)

            ##TODO: logger 로 이 print 값들 저장 + 하이퍼파라미터 값 저장
            
            #logger.write_logs(n_epi, n_step, average_reward, eff, maxeff, epilen, memory_size, ,epsilon,  )

            print("n_episode : {}, score : {}, eff : {}, effmax : {}, episode length : {}, n_buffer : {}, eps : {:.1f}%".format(
                  n_epi, score, eff_next, np.max(eff_epi_st), epi_length, memory.size(), epsilon*100))  #score/print_interval
            logger
            
            if args.tb==True:
                writer.add_scalar('efficency', eff_next, n_epi)
                writer.add_scalar('max efficiency', np.max(eff_epi_st), n_epi)
                writer.add_scalar('memory size', memory.size(), n_epi)
                writer.add_scalar('episode length', epi_length, n_epi)
            
            if args.save_devices == True:
                logger.deviceplotter(path_devices, s, n_epi)
            
            if args.save_np_struct == True: 
                logger.numpystructplotter(path_np_struct, s, n_epi)

        score = 0.0

    if args.save_summary == True:

        logger.summaryplotter(q, epi_len_st, s, path_summary)
    

    #logging   TO DO



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

    torch.save(q.state_dict(), q_net_name)
    torch.save(q_target.state_dict(), q_target_net_name)
    env.close()



