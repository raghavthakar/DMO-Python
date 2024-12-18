import math
import numpy

class BeachSection:
    def __init__(self):
        self.cap = 3
    
    def get_cap_reward(self, num_occupants: int):
        return num_occupants * math.exp(-num_occupants / self.cap)

    def get_mix_reward(self, occupants: list, num_beach_sections: int):
        return min(occupants) / (sum(occupants) * num_beach_sections)

class MOBeachEnv:
    def __init__(self):
        self.beach_sections = [BeachSection() for _ in range(5)]

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
            assert type(i) is int, "Tourist locations must be integers."
        for t in tourist_types:
            assert type(t) is int, "Tourist types must be integers."
            assert t in [0, 1], "Tourist type must be 0 or 1."
        assert len(tourist_locations) == len(tourist_types), "Number of tourists should match in locations and types."
        
        # Convert the agent locations to an occupation distribution
        # A 2 x num_beach_sections matrix (a row for type 0 and a row for type 1)
        tourist_distribution = numpy.zeros((len(self.beach_sections), 2), dtype=int)
        for beach_sec, tourist_type in zip(tourist_locations, tourist_types):
            tourist_distribution[beach_sec][tourist_type] += 1

        # Get the global rewards
        cap_reward = 0
        mix_reward = 0
        for i, beach_section in enumerate(self.beach_sections):
            cap_reward += beach_section.get_cap_reward(tourist_distribution[i])
            mix_reward += beach_section.get_mix_reward(tourist_distribution[i])
        
        return cap_reward, mix_reward

    def get_observations(self, tourist_locations):
        """
        Get the number of agents in the same beach section as each agent in the input.

        Parameters:
        - tourist_locations (list): List of tourist positions, each element being a non-negative integer.

        Returns:
        - observations (list): Number of agents in the same beach section as each agent in the input list.
        """
        # Convert the agent locations to an occupation distribution
        tourist_distribution = numpy.zeros(len(self.beach_sections), dtype=int)
        for beach_sec in tourist_locations:
            tourist_distribution[beach_sec] += 1
        # Create an observation vector by counting the total occupancy of the beach section occupied by each agent
        observations = []
        for i, beach_sec in enumerate(tourist_locations):
            observations.append(tourist_distribution[beach_sec])
        
        return observations