# COMP580-Project-Improving-DDQN-with-Count-based-Hashing
# RL Agent Training and Analysis

This repository provides an implementation of DDQN-based reinforcement learning agents, allowing users to train agents, perform ablation studies, and generate plots to analyze performance. Configurable hyperparameters enable flexibility for experimentation across various GYM environments.

## Setting Up the Environment

To set up the required dependencies, run the following command:
```
pip install gymnasium torch seaborn 
```

## Running the Script

### Configurations
All configurations are defined in the `utils/config.py` file, where you can set various hyperparameters for training. Key parameters include:
- **Environment Name** (* *env_name* *): Specify the Gymnasium environment.
- **Agent Class** (* *agent_class* *): Choose the agent implementation.
- **Number of Episodes** (* *n_episodes* *): Define how many episodes to train for.
- **Target Update Parameter** (* *target_update* *): Frequency for updating the target network.
- **Horizon** (* *horizon* *): Specify the maximum steps per episode, which depends on the environment.

Additional agent-specific parameters and environment-specific settings can also be adjusted in config.py.

## Training
To train the agent, make the necessary changes in  `config.py `and run:
```
python main.py
```
By default, the script uses 3 random seeds. You can add or remove seeds in `config.py` as needed.

## Plotting Results
After training, you can visualize the results by generating plots. Run
```
python utils/plot.py
```

## Ablation Studies
For ablation studies, modify the parameters in `config.py` and execute:
```
python main_ablation.py
```

## Contributions
Feel free to open issues or submit pull requests to improve the repository.
