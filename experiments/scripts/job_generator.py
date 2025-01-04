import os

def generate_bash_scripts(script_num, time, label, alg, env, data_dir, alg_config, env_config, seed):
    script_path = f"/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/experiments/scripts/job_scripts/{env}_{label}_{alg}_{seed}.sh"
    
    with open(script_path, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH --time={time}\n")
        file.write("#SBATCH --partition=share,dgx2,dgxh,ampere,mime1\n")
        file.write("#SBATCH --constraint=skylake\n")
        file.write("#SBATCH --mem=16G\n")
        file.write("#SBATCH -c 1\n\n")
        file.write("module load conda\n\n")
        file.write("source activate base\n\n")
        file.write("conda activate /nfs/stak/users/thakarr/hpc-share/GECCO25\n\n")
        file.write(f'/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/main.py "{alg}" "{env}" "{data_dir}" "{alg_config}" "{env_config}" "{seed}"\n')

    # Make the file executable
    os.chmod(script_path, 0o755)

    print(f"Generated script: {script_path}")

# Example usage:
time = "0-36:00:00"  # Set the time
env = "rover"
labels = ["coord4", "coord8", "coord12", "coord16"]
algs = ["nsga2", "dmo", "kpnsga2", "nsga2+d"]
data_dir="/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/experiments/data/"
alg_configs = [
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/4ag_3obj_12poi_DMOConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/8ag_3obj_12poi_DMOConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/12ag_3obj_12poi_DMOConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/16ag_3obj_12poi_DMOConfig.yaml",
    ]
env_configs = [
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/4ag_3obj_12poi_MORoverEnvConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/8ag_3obj_12poi_MORoverEnvConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/12ag_3obj_12poi_MORoverEnvConfig.yaml",
    "/nfs/stak/users/thakarr/hpc-share/GECCO25/DMO-Python/config/16ag_3obj_12poi_MORoverEnvConfig.yaml",
    ]
seeds = [2024, 2025, 2026, 2027]

for label, alg_config, env_config in zip(labels, alg_configs, env_configs):
    for alg in algs:
        for script_num, seed in enumerate(seeds):
            generate_bash_scripts(script_num, time, label, alg, env, data_dir, alg_config, env_config, seed)