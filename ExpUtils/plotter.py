import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Import IGD from pymoo
from pymoo.indicators.igd import IGD
# Import pygmo for finding the NDF
import pygmo as pg
# For SEM (standard error of the mean)
from scipy.stats import sem

# ----------------------------
# 1. Define Multiple Methods and Their CSV Files
# ----------------------------
methods_data = {
    "nsga2": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_beach_2024_beach50_2025-01-06_20-55-53_savedata.csv",
    ],
    "dmo": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_beach_2024_beach50_2025-01-06_20-22-00_savedata.csv",
    ],
    "kpnsga2": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_beach_2024_beach50_2025-01-06_22-15-26_savedata.csv",
    ],
    "nsga2+d": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_beach_2024_beach50_2025-01-06_23-03-35_savedata.csv",
    ],
    # Add more methods as needed
}

# ----------------------------
# 2. Known Pareto Front
# ----------------------------
# pareto_front = np.array([[-4, -4,  0],
#                          [-3, -4, -1],
#                          [-2, -4, -2],
#                          [-1, -4, -3],
#                          [ 0, -4, -4]])

pareto_front = np.array([[-4.107372, -0.452381],
                         [-4.134956, -0.450000],
                         [-4.162565, -0.447368],
                         [-4.190221, -0.444444],
                         [-4.217961, -0.441176],
                         [-4.239412, -0.415315],
                         [-4.267104, -0.412381],
                         [-4.288620, -0.385965],
                         [-4.316275, -0.383333],
                         [-4.337837, -0.356410],
                         [-4.365466, -0.354054],
                         [-4.414673, -0.324561]])

igd_indicator = IGD(pareto_front)

# ----------------------------
# 3. Helper Function: compute_gd_for_file
# ----------------------------
def compute_gd_for_file(csv_file):
    """
    Reads a CSV file, parses the 'fitness' column,
    computes the Non-dominated Front for each generation,
    and returns a dictionary: gen -> GD value.
    """
    df = pd.read_csv(csv_file, usecols=["gen", "fitness"])
    df['fitness_list'] = df['fitness'].apply(lambda x: ast.literal_eval(x))
    
    gd_dict = {}
    for g in sorted(df['gen'].unique()):
        gen_data = df[df['gen'] == g]['fitness_list'].tolist()
        ndf, _, _, _ = pg.fast_non_dominated_sorting(points=gen_data)
        ndf_fitnesses = np.array([gen_data[i] for i in ndf[0]])
        F = np.unique(ndf_fitnesses, axis=0)
        gd_val = igd_indicator(F)
        gd_dict[g] = gd_val
    return gd_dict

# ----------------------------
# 4. Plot Settings
# ----------------------------
plt.figure(figsize=(10, 6))

# Define a colormap or cycle through default colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# or manually define a dict of colors, e.g.:
# method_colors = {
#     "nsga2": "blue",
#     "dmo": "red",
#     ...
# }

# Keep track of each method index so we can pick a color
method_idx = 0

# ----------------------------
# 5. Loop Over Methods, Aggregate, and Plot
# ----------------------------
for method_name, csv_files in methods_data.items():
    
    if not csv_files:
        # Skip if there are no files for this method
        continue
    
    # Dictionary to collect generation -> [GD values across files]
    gd_per_generation = {}
    
    # Process each file for this method
    for file_path in csv_files:
        gd_dict = compute_gd_for_file(file_path)
        for gen, gd_val in gd_dict.items():
            if gen not in gd_per_generation:
                gd_per_generation[gen] = []
            gd_per_generation[gen].append(gd_val)
    
    # Compute mean and SEM
    sorted_gens = sorted(gd_per_generation.keys())
    mean_gds, sem_gds = [], []
    for gen in sorted_gens:
        values = gd_per_generation[gen]
        mean_gd = np.mean(values)
        sem_gd = sem(values)
        mean_gds.append(mean_gd)
        sem_gds.append(sem_gd)
    
    mean_gds = np.array(mean_gds)
    sem_gds = np.array(sem_gds)
    
    # Pick a color for this method
    color = colors[method_idx % len(colors)]
    method_idx += 1
    
    # Plot the mean
    plt.plot(sorted_gens, mean_gds, label=method_name, color=color)
    
    # Plot the shaded region for ±SEM
    plt.fill_between(sorted_gens,
                     mean_gds - sem_gds,
                     mean_gds + sem_gds,
                     alpha=0.2,
                     color=color)

# ----------------------------
# 6. Final Touches
# ----------------------------
plt.xlabel("Generation")
plt.ylabel("Generational Distance")
plt.title("Comparison of GD Across Multiple Methods")
plt.grid(True)
plt.legend()
plt.show()
