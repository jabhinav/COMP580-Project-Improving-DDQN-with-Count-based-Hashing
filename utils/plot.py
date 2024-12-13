import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


# Function to plot mean and standard deviation
def plot_training_logs(data, episodes, title, ylabel, save_at):
	"""
	Plot mean and standard deviation of training logs across multiple runs for an algorithm.
	:param data:
	:param episodes:
	:param title:
	:param ylabel:
	:param save_at:
	:return:
	"""
	data = np.array(data)
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	
	sns.set(style="whitegrid")
	plt.figure(figsize=(10, 6))
	
	plt.plot(episodes, mean, label=f"Mean {title}")
	plt.fill_between(episodes, mean - std, mean + std, alpha=0.3)
	
	plt.title(f"Plot of {title} across training runs")
	plt.xlabel("Episode")
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(os.path.join(save_at, f"{title}.png"), dpi=300)
	
	
def compare_plots_of_training_logs(data, episodes, title, ylabel, save_path):
	"""
	Plot and compare mean and standard deviation of training logs across multiple runs for n algorithms.
	:param data: Dict of data for each algorithm with key as algorithm name and value as list of logs across runs.
	:param episodes:
	:param title:
	:param ylabel:
	:param save_at:
	:return:
	"""
	mean = {}
	std = {}
	colors = sns.color_palette("deep", n_colors=len(data.keys()))
	for alg in data.keys():
		data[alg] = np.array(data[alg])
		mean[alg] = np.mean(data[alg], axis=0)
		std[alg] = np.std(data[alg], axis=0)
		
	sns.set(style="whitegrid")
	plt.figure(figsize=(14, 8))
	
	for alg in data.keys():
		plt.plot(episodes, mean[alg], label=f"{alg}", color=colors.pop(0))
		plt.fill_between(episodes, mean[alg] - std[alg], mean[alg] + std[alg], alpha=0.2)
		
	plt.title(f"{title}")
	plt.xlabel("Episode")
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(save_path, dpi=300)


def compute_running_avg(data, window=100):
	"""
	Compute running average of data.
	:param data:
	:param window:
	:return:
	"""
	running_avg = []
	for i in range(len(data)):
		if i < window:
			running_avg.append(np.mean(data[:i + 1]))
		else:
			running_avg.append(np.mean(data[i - window:i]))
	return running_avg

	
def read_data_from_csv(path_to_csv, cols_to_read=["Total_Reward"]):
	"""
	Read data from a CSV file and return as a dictionary.
	:param path_to_csv:
	:return:
	"""
	data = {}
	df = pd.read_csv(path_to_csv)
	for col in df.columns:
		if col in cols_to_read:
			# data[col] = df[col].values
			data[col] = compute_running_avg(df[col].values)
	return data


def read_all_data_for_algorithm(path_to_dir):
	"""
	Read all CSV files in a directory and return as a dictionary.
	:param path_to_dir:
	:return: Dict with keys as metric names and values as list of lists of values across runs.
	"""
	data = {}
	for file in os.listdir(path_to_dir):
		if file.endswith(".csv"):
			file_path = os.path.join(path_to_dir, file)
			run_data = read_data_from_csv(file_path)
			for key, value in run_data.items():
				if key not in data:
					data[key] = []
				data[key].append(value)
				
	return data


def read_all_algorithm_results(path_to_alg_results: dict):
	"""
	Read all results for multiple algorithms and return as a dictionary.
	:param path_to_alg_results: Dict with keys as algorithm names and values as path to directory containing results.
	:return: Dict with keys as algorithm names and values as dict of metric names and values across runs.
	"""
	data = {}
	for alg, path in path_to_alg_results.items():
		data[alg] = read_all_data_for_algorithm(path)
	return data


def main():
	# Set path to results directory
	results_dir = "../results/Ablation/alpha_count"
	save_path = os.path.join(results_dir, "ablation_count_based_alpha_runningAvg.png")
	
	# # Read all results for multiple algorithms
	# alg_results = {
	# 	"Baseline": os.path.join(results_dir, "DDQNAgent_CartPole-v1_baseline"),
	# 	"CountBased": os.path.join(results_dir, "DDQNAgent_CartPole-v1_count_based"),
	# 	"PER": os.path.join(results_dir, "DDQNAgent_CartPole-v1_per"),
	# 	"CountBased_w_PER": os.path.join(results_dir, "DDQNAgent_CartPole-v1_count_based_per"),
	# }
	
	alg_results = {
		"alpha=0.1": os.path.join(results_dir, "DDQNAgent_CartPole-v1_alpha0.1_count_based"),
		"alpha=0.2": os.path.join(results_dir, "DDQNAgent_CartPole-v1_alpha0.2_count_based"),
		"alpha=0.4": os.path.join(results_dir, "DDQNAgent_CartPole-v1_alpha0.4_count_based"),
		"alpha=0.8": os.path.join(results_dir, "DDQNAgent_CartPole-v1_alpha0.8_count_based"),
	}
	data = read_all_algorithm_results(alg_results)
	
	# # Plot and compare training logs
	metric_to_plot = 'Total_Reward'
	metric_data = {}
	for alg in data.keys():
		metric_data[alg] = data[alg][metric_to_plot]
	episodes = np.arange(1, 1001)
	compare_plots_of_training_logs(metric_data, episodes, title="Performance Comparison", ylabel="Reward",
									save_path=save_path)


if __name__ == "__main__":
	main()