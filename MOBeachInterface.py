import yaml
import torch
import numpy as np

from Policy import Policy
from MOBeachEnv import MOBeachEnv

class MOBeachInterface():
    def __init__(self, beach_config_filename):
        """
        Initialise the MOROverInterface class with its instance of the MOBeachEnv Domain.
        Setup an internal reference to the rover config file
        """
        self.beach_env = MOBeachEnv(beach_config_filename)
        with open(beach_config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
    
    # to perform a key-wise sum of two dicts
    def _keywise_sum(self, dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

    def rollout(self, joint_policy: list):
        """
        Perform a rollout of a given multiheaded actor in the MOBeachEnv domain.

        Parameters:
        - joint_policy: list of policies, representing the whole team

        Returns:
        - rollout_trajectory (dict): Complete trajectory of the rollout with position, and action data of each agent.
        - global_reward (list): Reward vector that evaluates this joint_policy on each system-level objective.
        """
        if not (isinstance(joint_policy, list) and isinstance(p, Policy) for p in joint_policy):
            raise ValueError("The supplied joint policy should be a list of Policy type objects")

        ep_length = self.beach_env.get_ep_length()

        # Process the starting agent distribution to set agent locations
        agent_locations = []
        agent_types = []
        for i, section in enumerate(self.config['Environment']['sections']):
            # Add all type-0 agents for this section
            for _ in range(section['num_type0_agents']):
                agent_locations.append(i)
                agent_types.append(0)
                
            # Add all type-1 agents for this section
            for _ in range(section['num_type1_agents']):
                agent_locations.append(i)
                agent_types.append(1)
        
        cumulative_global_reward = {}  # Initialize cumulative global reward

        rollout_trajectory = [[] for _ in range(len(joint_policy))] # List of list of dicts

        for t in range(ep_length):
            # get each agent's observation at the current position
            observations_list = self.beach_env.generate_observations(agent_locations) 
            
            joint_action = [] # Will store the combined joint action

            # get each agent's move based on corresponding observation
            for i, policy in enumerate(joint_policy):
                # extract singe agent's observation
                agent_observation = observations_list[i]
                agent_observation_tensor = torch.FloatTensor(agent_observation)
                # get policy output based on observation
                action_dist_tensor = policy.forward(agent_observation_tensor, final_activation="softmax")
                 # Convert to a numpy array without tracking gradient
                action_dist_vector = action_dist_tensor.squeeze(0).detach().numpy()
                # Pick the action that has the highest value
                action = np.argmax(action_dist_vector) - 1 # -1, 0, or +1

                # Add scaled action to the list of agent moves
                joint_action.append(action)

                # Add the agent's transition to the trajectory
                rollout_trajectory[i].append(
                    {
                        'state' : agent_observation,
                        'action' : action,
                        'position': agent_locations[i],  # Store the actual position
                        'type': agent_types[i],
                    }
                )

            # get updated agent positions based on the joint action
            agent_locations = self.beach_env.update_agent_locations(tourist_locations=agent_locations, 
                                                                    tourist_deltas=joint_action)
            
            # Get the global reward and update the cumulative global reward
            global_reward = self.beach_env.get_global_rewards(tourist_locations=agent_locations, tourist_types=agent_types)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)

        return rollout_trajectory, cumulative_global_reward

    # Function that evaluates a given trajectory for global rewards (without rollout)
    def evaluate_trajectory(self, traj: dict):
        parsed_trajectory = [
            ([agent['position'] for agent in timestep], [agent['type'] for agent in timestep])
            for timestep in zip(*traj)
        ] # NOTE: this only works for 2-dimensional environments

        cumulative_global_reward = {}  # Initialize cumulative global reward
        for t, (agent_locations, agent_types) in enumerate(parsed_trajectory):
            global_reward = self.beach_env.get_global_rewards(tourist_locations=agent_locations, tourist_types=agent_types)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)
        
        return cumulative_global_reward

    # Function to get domain-specific information for the algorithm
    def get_state_size(self):
        '''
        Get the number of inputs to the actor network.
        '''
        return self.beach_env.num_beach_sections
    
    def get_action_size(self):
        '''
        Get the number of outputs per agent for the actor network.
        '''
        return 3

    def get_team_size(self):
        '''
        Get the number of agents that must be deployed in a rollout. Team size.
        '''
        # team size is implicitly recorded by the length of starting locations
        return self.beach_env.get_num_agents()
    
    def get_num_objs(self):
        '''
        Get the number of objectives in the problem.
        '''
        return self.beach_env.get_num_objs()