import yaml
import pygmo as pg
import random
import torch
import numpy

import MORoverInterface
import Individual
import Utils

import ExpUtils.DataLogger

class CentralisedAlgorithm:
    def __init__(self, alg_config_filename, rover_config_filename, data_filename):
        self.config_filename = alg_config_filename
        self._read_config()

        self.data_filename = data_filename
        self.data_logger = ExpUtils.DataLogger.DataLogger(data_fields=['gen',
                                                                       'id',
                                                                       'fitness',
                                                                       'trajectory'],
                                                                       target_filename=self.data_filename)

        self.interface = MORoverInterface.MORoverInterface(rover_config_filename)
        self.team_size = self.interface.get_team_size()
        self.num_objs = self.interface.get_num_objs()

        self.pop = []
        self.glob_ind_counter = 0
        
        # Create the initial population
        for i in range(self.pop_size):
            # Add new individual ot the population
            self.pop.append(Individual.Individual(config_filename=self.config_filename,
                                                  num_agents=self.team_size,
                                                  input_size=self.interface.get_state_size(),
                                                  output_size=self.interface.get_action_size(),
                                                  id=i,
                                                  num_objs=self.num_objs))
            # Increment the global id counter
            self.glob_ind_counter += 1
        
        # Evo utils
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
