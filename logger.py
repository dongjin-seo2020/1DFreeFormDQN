import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import os

#write logs and .csv file of hyperparmeters
def write_logs(loggername, lgr, sh, n_epi, eff, effmax, \
            episode_length, n_buffer, epsilon_percent, count):
    
    # create logger - saving folder: tb folder
    lgr.setLevel(logging.DEBUG)
    lgr.info('n_epi: %s, eff: %s, effmax: %s, episode_length: %s, n_buffer: %s, epsilon_percent:%s, count:%s',\
                n_epi, eff, effmax, episode_length, n_buffer, epsilon_percent, count)


    #csv
    output = csv.writer(open(loggername+'.csv', 'w'))
    output.writerow(['time', 'n_epi', 'eff', 'effmax', 'episode_length', 'n_buffer', 'epsilon [%]', 'count'])
    output.writerow([time.strftime('%Y_%m_%d %H:%M:%S'), n_epi, eff, effmax, episode_length, n_buffer, epsilon_percent, count])
    #csv

    if os.path.isfile(loggername+'.csv'):
        #with open(loggername+'.csv',newline='') as f:
         #   r = csv.reader(f)
          #  data = [line for line in r]
        with open(loggername+'.csv','a',newline='') as f:
            w = csv.writer(f)
            #w.writerow(['time', 'n_epi', 'eff', 'effmax', 'episode_length', 'n_buffer', 'epsilon [%]'])
            #w.writerow(data)
            w.writerow([time.strftime('%Y_%m_%d %H:%M:%S'), n_epi, eff, effmax, episode_length, n_buffer, epsilon_percent, count])
  
    else:
        output = csv.writer(open(loggername+'.csv', 'w', newline=''))
    	output.writerow(['time', 'n_epi', 'eff', 'effmax', 'episode_length', 'n_buffer', 'epsilon [%]', 'count'])
    	output.writerow([time.strftime('%Y_%m_%d %H:%M:%S'), n_epi, eff, effmax, episode_length, n_buffer, epsilon_percent, count])
    

def write_json_hyperparameter(path_logs_tb, args):
    #write json file of hyperparameters
    with open(path_logs_tb+'/config.json','w') as fp:
        print(vars(args))
        json.dump(vars(args),fp)

def final_logs(eff_init, eff_next, q, q_target):
    pass

def summaryplotter(q, epi_len_st, struct, path_summary):
    plt.figure()
    plt.plot(q.effdata)
    plt.title('eff')
    plt.savefig(path_summary+'_eff', format='eps')
    plt.close()

    plt.figure()
    plt.plot(epi_len_st)
    plt.title('episode length')
    plt.savefig(path_summary+'_epi_length', format='eps')
    plt.close()

    plt.figure(figsize=(20,10))
    plt.imshow(struct.reshape(1,-1),cmap='Greys')
    plt.yticks([])
    plt.title('final structure')
    plt.savefig(path_summary+'_structure', format='eps')
    plt.close()

def deviceplotter(path_devices, struct, n_epi):
    path_devices = path_devices.format(n_epi)
    plt.figure(figsize=(20,10))
    plt.imshow(struct.reshape(1,-1),cmap='Greys')
    plt.yticks([])
    plt.savefig(path_devices)
    plt.close() 
    
def numpystructplotter(path_np_struct, struct, n_epi):
    path_np_struct = path_np_struct.format(n_epi)
    np.save(path_np_struct, struct)
    
