import random
import torch
import numpy

import NSGAII
import KParentNSGAII
import DMO

random.seed(2024)
torch.manual_seed(2024)
numpy.random.seed(2024)

if __name__ == '__main__':
    alg = NSGAII.NSGAII(alg_config_filename='/home/raghav/Research/GECCO25/DMO/config/DMOConfig.yaml',
                 data_filename='/home/raghav/Research/GECCO25/DMO/experiments/data/testing.csv',
                 rover_config_filename='/home/raghav/Research/GECCO25/DMO/config/MORoverEnvConfig.yaml')