import random
import pygmo as pg

import Algorithm

class NSGAII(Algorithm.Algorithm):
    def evolve(self):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Condcut rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)
            if len(fitness_dict) != self.num_objs:
                raise ValueError(f"[NSGA-II] Expected {self.num_objs} objectives, but got {len(fitness_dict)}.")
            # Store the rollout trajectory
            ind.trajectory = trajectory
            # Store fitness
            for f in fitness_dict:
                ind.fitness[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
        
        # Sort the population according to fitness
        sorted_indices = pg.sort_population_mo(points=[ind.fitness for ind in self.pop])
        # print(sorted_indices)
        for i in sorted_indices:
            print(self.pop[i])
        print("----------------------------")
        # Save only the top half of the population
        sorted_indices = sorted_indices[:len(sorted_indices)//2]

        # Save these elites into the parent set
        parent_set = [self.pop[elite_idx] for elite_idx in sorted_indices]

        # Create empty offpring set
        offspring_set = []

        # Fill up the offspring set to the pop_size via offspring-creation
        while len(parent_set) + len(offspring_set) < self.pop_size:
            # Select 2 parents via binary tournament
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent1 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent2 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            # Get the offsprings by crossing over these Individuals
            offspring1, offspring2 = self.utils.crossover(parent1, parent2)
            # Mutate the offsprings by adding noise
            offspring1.mutate()
            offspring2.mutate()
            # Add to the offspring set
            offspring_set.extend([offspring1, offspring2])
        
        # Set the population to the parent + offspring set
        self.pop = parent_set
        self.pop.extend(offspring_set)

        random.shuffle(self.pop) # NOTE: This is so that equally dominnat offpsrings in later indices don't just get thrown out


nsga = NSGAII()

for _ in range(10000):
    nsga.evolve()
