import yaml
import pygmo as pg

import MORoverInterface
import Individual

class NSGAII:
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
    
    def evolve(self):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Condcut rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)
            if len(fitness_dict) != self.num_objs:
                raise ValueError("[NSGA-II] The fitness vector from the rollout should have as many elements as number of objectives.")
            # Store fitness
            for f in fitness_dict:
                ind.fitness[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
        
        # Sort the population according to fitness
        sorted_indices = pg.sort_population_mo(points=[ind.fitness for ind in self.pop])
        print(sorted_indices)


nsga = NSGAII()
nsga.evolve()
