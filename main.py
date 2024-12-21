import random
import torch
import numpy
import sys
import datetime
import shutil # for file management

import NSGAII
import KParentNSGAII
import DMO
import NSGAII_D

if __name__ == '__main__':
    assert len(sys.argv) == 7, "Correct usage: python alg_name domain_name data_dirpath alg_config env_config seed"
   
    # Process the command line args
    alg_name = sys.argv[1]
    assert alg_name in ['nsga2', 'kpnsga2', 'dmo', 'nsga2+d'], "Unrecognised alg_name"
    domain_name = sys.argv[2]
    assert domain_name in ['rover', 'beach'], 'Uncrecognised domain_name'
    data_dir = sys.argv[3]
    data_dir = data_dir+'/' if data_dir[-1]!='/' else data_dir # Add a directory '/' at the end
    src_alg_config_filename = sys.argv[4]
    src_env_config_filename = sys.argv[5]
    seed_val = int(sys.argv[6])

    # Datetime for file naming
    datetime_now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save data filename
    data_filename = data_dir+alg_name+'_'+datetime_now_string+'_savedata.csv'
    # Create copy of configs at save data location
    dest_alg_config_filename = data_dir+alg_name+'_'+domain_name+'_'+datetime_now_string+'_algconfig.yaml'
    shutil.copyfile(src_alg_config_filename, dest_alg_config_filename)
    dest_env_config_filename = data_dir+alg_name+'_'+domain_name+'_'+datetime_now_string+'_envconfig.yaml'
    shutil.copyfile(src_env_config_filename, dest_env_config_filename)

    # Set the seed value for all libraries
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    numpy.random.seed(seed_val)

    # Initialise the algorithm based on alg name
    if alg_name == 'nsga2':
        alg = NSGAII.NSGAII(alg_config_filename=dest_alg_config_filename,
                            domain_name=domain_name,
                            data_filename=data_filename,
                            rover_config_filename=dest_env_config_filename)
    elif alg_name == 'kpnsga2':
        alg = KParentNSGAII.KParentNSGAII(alg_config_filename=dest_alg_config_filename,
                                          domain_name=domain_name,
                                          data_filename=data_filename,
                                          rover_config_filename=dest_env_config_filename)
    elif alg_name == 'dmo':
        alg = DMO.DMO(alg_config_filename=dest_alg_config_filename,
                      domain_name=domain_name,
                      data_filename=data_filename,
                      rover_config_filename=dest_env_config_filename)
    elif alg_name == 'nsga2+d':
        alg = NSGAII_D.NSGAII_D(alg_config_filename=dest_alg_config_filename,
                                domain_name=domain_name,
                                data_filename=data_filename,
                                rover_config_filename=dest_env_config_filename)
    
    # Run the algorithm
    for gen in range(alg.num_gens):
        alg.evolve(gen=gen)