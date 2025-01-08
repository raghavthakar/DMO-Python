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
        "~/gecco_data/reduced_nsga2_rover_2024_coord16_2025-01-04_15-19-56_savedata.csv",
        "~/gecco_data/reduced_nsga2_rover_2025_coord16_2025-01-04_15-20-59_savedata.csv",
        "~/gecco_data/reduced_nsga2_rover_2026_coord16_2025-01-04_15-21-56_savedata.csv",
        "~/gecco_data/reduced_nsga2_rover_2027_coord16_2025-01-04_15-22-57_savedata.csv",
    ],
    "dmo": [
        "~/gecco_data/reduced_dmo_rover_2024_coord16_2025-01-04_15-11-57_savedata.csv",
        "~/gecco_data/reduced_dmo_rover_2025_coord16_2025-01-04_15-12-58_savedata.csv",
        "~/gecco_data/reduced_dmo_rover_2026_coord16_2025-01-04_15-13-55_savedata.csv",
        "~/gecco_data/reduced_dmo_rover_2027_coord16_2025-01-04_15-14-55_savedata.csv",
    ],
    "kpnsga2": [
        "~/gecco_data/reduced_kpnsga2_rover_2024_coord16_2025-01-04_15-15-55_savedata.csv",
        "~/gecco_data/reduced_kpnsga2_rover_2025_coord16_2025-01-04_15-16-58_savedata.csv",
        "~/gecco_data/reduced_kpnsga2_rover_2026_coord16_2025-01-04_15-17-55_savedata.csv",
        "~/gecco_data/reduced_kpnsga2_rover_2027_coord16_2025-01-04_15-18-56_savedata.csv",
    ],
    "nsga2+d": [
        "~/gecco_data/reduced_nsga2+d_rover_2024_coord16_2025-01-04_15-23-57_savedata.csv",
        "~/gecco_data/reduced_nsga2+d_rover_2025_coord16_2025-01-04_15-25-00_savedata.csv",
        "~/gecco_data/reduced_nsga2+d_rover_2026_coord16_2025-01-04_15-27-01_savedata.csv",
        "~/gecco_data/reduced_nsga2+d_rover_2027_coord16_2025-01-04_15-27-58_savedata.csv",
    ],
    # Add more methods as needed
}

# ----------------------------
# 2. Known Pareto Front
# ----------------------------
pareto_front = np.array([[-4, -4,  0],
                         [-3, -4, -1],
                         [-2, -4, -2],
                         [-1, -4, -3],
                         [ 0, -4, -4]])

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
