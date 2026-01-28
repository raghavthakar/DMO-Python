# Supporting Code for the DMO Operator for multi-objective multiagent credit assignment

Raghav Thakar, Gaurav Dixit, Siddarth Iyer, and Kagan Tumer. 2025. Multiagent Credit Assignment for Multi-Objective Coordination. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '25). Association for Computing Machinery, New York, NY, USA, 663–672. https://doi.org/10.1145/3712256.3726445

## Summary
This code base contains all the code used in the paper, from the simulation environment to the main contribution, and the baselines as well.

## Instructions
Run `main.py` with the desired arguments to launch your choice of algorithm in your choice of environment.

Define parameters for your algorithm and environment in a `.yaml` inside `config/`.

Use `ExpUtils/RoverEnvViz.py` to visualise your config of the Multi-Objective Rover Domain Environment.

Example usage:
```
python main.py dmo rover ~/dmo_rover_data/ \
config/concentric/8ag_concentric_DMOConfig.yaml \
config/concentric/8ag_concentric_MORoverEnvConfig.yaml \
2024 dmo_rover_test 5
```

## Avilable algorithms:
- nsga2: Plain NSGA-2
- dmo: Main contribution of the paper, implements credit assignment with a modified NSGA-2 algorithm (K-parent crossover)
- kpnsga2: Ablated version with no credit assignment, but still performs the K-parent crossover
- nsga2+d: Baseline from the literature

### Original baseline paper:
Yliniemi, L., Tumer, K. (2014). Multi-objective Multiagent Credit Assignment Through Difference Rewards in Reinforcement Learning. In: Dick, G., et al. Simulated Evolution and Learning. SEAL 2014.

## Some more thoughts
If you find this repository useful, please consider citing our work:

Raghav Thakar, Gaurav Dixit, Siddarth Iyer, and Kagan Tumer. 2025. Multiagent Credit Assignment for Multi-Objective Coordination. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '25). Association for Computing Machinery, New York, NY, USA, 663–672. https://doi.org/10.1145/3712256.3726445

Maintained by Raghav Thakar. Please raise an issue if you have an issues using this repository.
