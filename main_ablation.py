from collections import defaultdict
from train import train_rl_agent
from utils.plot import plot_training_logs
import numpy as np
import os
import utils.config as cfg
import pandas as pd
import random
import torch


# Parameter over which ablation study is to be performed
ablation_var = "beta"
ablation_values = [0.1, 0.5, 1.0]
seeds = [0, 1, 2]

for value in ablation_values:
	
	# Update config
	setattr(cfg, ablation_var, value)
	
	# Save results
	save_at = os.path.join(f"results/{cfg.agent_class.__name__}_{cfg.env_name}_{ablation_var}{value}")
	if not cfg.agent_args["count_based"] and not cfg.agent_args["use_per"]:
		save_at += "_baseline"
	else:
		if cfg.agent_args["count_based"]:
			save_at += "_count_based"
		if cfg.agent_args["use_per"]:
			save_at += "_per"
	
	if not os.path.exists(save_at):
		os.makedirs(save_at)
	
	results_cols = ["Episode", "Total_Reward", "Total_Steps", "Policy_Loss", "Epsilon"]
	results_df = pd.DataFrame(columns=results_cols)
	
	results = defaultdict(list)
	for seed in seeds:
		# Set seed for reproducibility
		np.random.seed(seed)
		random.seed(seed)
		torch.manual_seed(seed)
		
		results_dir = os.path.join(save_at, f"seed_{seed}")
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		
		# Train the agent
		logs = train_rl_agent(
			cfg.env_name,
			cfg.agent_class,
			cfg.agent_args,
			cfg.n_episodes,
			cfg.horizon,
			cfg.target_update,
			results_dir
		)
		for key, value in logs.items():
			results[key].append(value)
		
		# Save results to CSV
		for idx in logs["episode"]:
			results_df.loc[idx, "Episode"] = logs["episode"][idx]
			results_df.loc[idx, "Total_Reward"] = logs["rewards"][idx]
			results_df.loc[idx, "Total_Steps"] = logs["steps"][idx]
			results_df.loc[idx, "Policy_Loss"] = logs["loss"][idx]
			results_df.loc[idx, "Epsilon"] = logs["epsilon"][idx]
			results_df.to_csv(results_dir + "_training.csv")
		print("CSV saved ...")
	
	episodes = np.arange(1, len(results["rewards"][0]) + 1)
	for key, value in results.items():
		plot_training_logs(value, episodes, title=key, ylabel=key, save_at=save_at)
