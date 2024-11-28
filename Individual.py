import yaml

import Policy

class Individual:
    def __init__(self, 
                 config_filename, 
                 num_agents=10, 
                 input_size=10, 
                 output_size=2, 
                 id=-1, 
                 num_objs=2):
        # self._read_config(alg_config_filename)
        self.joint_policy = [Policy.Policy(config_filename, 
                                           input_size=input_size, 
                                           output_size=output_size) for _ in range(num_agents)]
        self.id = id
        self.num_objs = 2

        self.reset_fitness()
    
    def reset_fitness(self):
        """Zero the fitness of the individual."""
        self.fitness = [-1 for _ in range(self.num_objs)]