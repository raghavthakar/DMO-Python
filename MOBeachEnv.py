import math
import numpy
import yaml

class BeachSection:
    def __init__(self, cap):
        self.cap = cap
    
    def get_cap_reward(self, num_occupants: int):
        return num_occupants * math.exp(-num_occupants / self.cap)

    def get_mix_reward(self, occupants: list, num_beach_sections: int):
        return min(occupants) / (sum(occupants) * num_beach_sections) if sum(occupants) > 0 else 0

class MOBeachEnv:
    def __init__(self, config_filename):
        if not isinstance(config_filename, str):
            raise ValueError('Beach configuration filename must be a string.')

        self.config_filename = config_filename
        self._read_config()  # Initial loading of the environment configuration
        x = 2

    def _read_config(self):
        """Read and load environment configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[MOBeachEnv]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal environment configuration."""
        # Initialize environment properties
        self.num_objs = 2
        self.num_beach_sections = len(self.config_data['Environment']['sections'])
        self.ep_length = self.config_data['Environment']['ep_length']
        self.beach_sections = [BeachSection(section['capacity']) for section in self.config_data['Environment']['sections']]
        self.num_agents = 0
        for section in self.config_data['Environment']['sections']:
            self.num_agents = self.num_agents + section['num_type0_agents'] + section['num_type1_agents']

    def get_global_rewards(self, tourist_locations, tourist_types):
        """
        Calculate and return the net reward vector for a list of tourist positions.
        
        Parameters:
        - tourist_locations (list): List of tourist positions, each element being a non-negative integer.
        - tourist_types (list): List of tourist types, each element being a 0 or 1
        
        Returns:
        - reward_vector (dict): Dictionary where keys are objectives (obj) and values are rewards for each objective.
        """
        for i in tourist_locations:
            if not isinstance(i, (int, numpy.int16, numpy.int32, numpy.int64)):
                raise ValueError("Tourist locations must be integers.")
        for t in tourist_types:
            if not isinstance(t, (int, numpy.int16, numpy.int32, numpy.int64)):
                raise ValueError("Tourist types must be integers.")
            assert t in [0, 1], "Tourist type must be 0 or 1."
        assert len(tourist_locations) == len(tourist_types), "Number of tourists should match in locations and types."
        
        # Convert the agent locations to an occupation distribution
        # A num_beach_sections x 2 matrix (a row for each section)
        tourist_distribution = numpy.zeros((len(self.beach_sections), 2), dtype=int)
        for beach_sec, tourist_type in zip(tourist_locations, tourist_types):
            tourist_distribution[beach_sec][tourist_type] += 1

        # Get the global rewards
        cap_reward = 0
        mix_reward = 0
        for i, beach_section in enumerate(self.beach_sections):
            cap_reward += beach_section.get_cap_reward(sum(tourist_distribution[i]))
            mix_reward += beach_section.get_mix_reward(tourist_distribution[i], self.num_beach_sections)
        
        return {0 : cap_reward, 1 : mix_reward}
    
    def generate_observations(self, tourist_locations):
        """
        Generate a one-hot encoding of the beach section each agent is in.
        Each observation will be a list of length self.num_beach_sections,
        with a 1 at the index corresponding to the agent's current section and 0 elsewhere.
        """
        observations = []
        for section in tourist_locations:
            # Use a list comprehension with a conditional expression
            agent_observation = [1 if i == section else 0 for i in range(self.num_beach_sections)]
            observations.append(agent_observation)

        return observations

    
    def update_agent_locations(self, tourist_locations, tourist_deltas):
        """
        Update the locations of agents based on their moves.

        Parameters:
        - tourist_locations (list[int]): Current positions of each tourist (0-based index of beach sections).
        - tourist_deltas (list[int]): Movement decisions for each tourist (-1 for left, 0 for stay, +1 for right).

        Returns:
        - new_locations (list[int]): Updated positions of each tourist after applying the moves (clamped to valid range).
        """
        new_locations = []
        for loc, move in zip(tourist_locations, tourist_deltas):
            new_loc = loc + move
            if not isinstance(new_loc, (int, numpy.int16, numpy.int32, numpy.int64)):
                raise ValueError("The move caused agent location to be a float. Invalid. Exiting...")
            # Check if new_loc is an integer and within the valid range
            if 0 <= new_loc < self.num_beach_sections:
                new_locations.append(new_loc)
            else:
                # If not valid, do not move the tourist
                new_locations.append(loc)

        return new_locations
    
    def get_ep_length(self):
        return self.ep_length

    def get_num_objs(self):
        return self.num_objs

    def get_num_agents(self):
        return self.num_agents


if __name__ == "__main__":
    beach = MOBeachEnv('/home/raghav/Research/GECCO25/DMO/config/MOBeachEnvConfig.yaml')
    pos = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    types = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

    print(beach.get_global_rewards(pos, types))