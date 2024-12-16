import yaml
import pygmo as pg
import random
import torch
import numpy

import MORoverInterface
import Individual
import Utils

random.seed(2024)
torch.manual_seed(2024)
numpy.random.seed(2024)

class Algorithm:
    def __init__(self):
        self.config_filename = '/home/raghav/Research/GECCO25/DMO/config/DMOConfig.yaml'
        self._read_config()

        self.interface = MORoverInterface.MORoverInterface('/home/raghav/Research/GECCO25/DMO/config/MORoverEnvConfig.yaml')
        self.team_size = self.interface.get_team_size()
        self.num_objs = self.interface.get_num_objs()

        self.pop = [Individual.Individual(config_filename=self.config_filename, 
                                          num_agents=self.team_size, 
                                          input_size=self.interface.get_state_size(), 
                                          output_size=self.interface.get_action_size(), 
                                          id=1, 
                                          num_objs=self.num_objs) for _ in range(self.pop_size)]
        
        self.utils = Utils.Utils(num_objs=self.num_objs)
        
    def _read_config(self):
        """Read and load NSGA-II configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[NSGA-II]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal NSGA-II configuration."""
        self.pop_size = self.config_data['Evolutionary']['pop_size']
        self.num_gens = self.config_data['Evolutionary']['num_gens']
