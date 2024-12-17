import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Import GD from pymoo
from pymoo.indicators.gd import GD
# Import pygmo for finding the ndf
import pygmo as pg

# --- User Input Section ---
csv_file = "/home/raghav/Research/GECCO25/DMO/experiments/data/testing.csv"  # Replace with your actual CSV file path

# Define your known Pareto front here
pareto_front = np.array([[-4, -8, -4],
                         [-8, -8, 0],
                         [0, -8, -8]])

# Initialize the GD indicator with the known PF
gd_indicator = GD(pareto_front)

# Read CSV
df = pd.read_csv(csv_file)

# Parse the fitness column from a string representation of a list to a Python list
df['fitness_list'] = df['fitness'].apply(lambda x: ast.literal_eval(x))

# Get unique generations sorted
generations = sorted(df['gen'].unique())

gd_values = []
for g in generations:
    # Extract all fitness vectors for this generation
    gen_data = df[df['gen'] == g]['fitness_list'].tolist()
    # Find the Nondominated Front at this generation
    ndf, _, _, _ = pg.fast_non_dominated_sorting(points = gen_data)
    ndf_fitnesses = np.array([gen_data[i] for i in ndf[0]])
    # Get all the unique fitnesses in the nondominated front
    F = np.unique(ndf_fitnesses, axis=0)
    print(F)
    # Compute GD for this generation
    gd_val = gd_indicator(F)
    gd_values.append((g, gd_val))

# Separate generations and their GD values
gens, gds = zip(*gd_values)

# Plotting the generational distance
plt.figure(figsize=(10, 6))
plt.plot(gens, gds)
plt.xlabel("Generation")
plt.ylabel("Generational Distance")
plt.title("Generational Distance Over Generations Using pymoo")
plt.grid(True)

plt.show()
