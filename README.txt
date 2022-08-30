REPOSITORY FOR DEEP REINFORCEMENT LEARNING COURSEWORK

*******************************************************************************************************

This directory contains repositories for each task in the DRL coursework: BASIC, ADVANCED and ATARI

*******************************************************************************************************
DIRECTORY GUIDE

*******************************************************************************************************

BASIC:

1. agent_tools.py: Python file containing code for the 'Q_learner' class applied to Cartpole v1 
and the 'performance_splitter' function to get median agent performance

2. DRL.basic.optimised.ipynb: Jupyter notebook containing the trained optimised Q-learning agent 
for cartpole

3. DRL.basic.optuna.4.0.ipynb: Jupyter notebook containing the Optuna study evaluating hyperparameters for 
the Q-learning agent

4. DRL.basic.vis.0.0.ipynb: Jupyter notebook containing data visulisations for different Q-learning policies 
trained on Cartpole-v1

5. DRL.discrete.4.0.ipynb: Jupyter notebook containing discrete hyperparameter evaluations and visualisations for 
reference model trained on Cartpole-v1

6. vid_tools.py: Python file containing functions for recording the Q-learning agents with the reference 
used to create the code

7. basic_results: directory containing csv files for results from discrete experiments

8. optuna_results: directory containing the csv file with results from the optuna study

*******************************************************************************************************

ADVANCED

code:

	1. DRL.advanced.vis.0.0.ipynb: Jupyter notebook containing data visuliastions for different DQN models
	trained on Cartpole-v1

	2. LunLand.py: Python file containing the agent class to be trained on LunarLander

	3. memory.py: Python file containing experience replay for agents

	4. models.py: Python file containing NoisyLayer, DQN and Dueling DQN classes 
	(NoisyLayer was not used in experiments)

	5. opt_double_LL.ipynb: Jupyter notebook containing the trained optimised double DQN model 
	at 1000 timesteps per epsiode on LunarLander-v2

	6. opt_duel_double_LL.ipynb: Jupyter notebook containing the trained optimised dueling double 
	DQN model at 1000 timesteps per epsiode on LunarLander-v2

	7. opt_duel_double_half_LL.ipynb: Jupyter notebook containing the trained optimised dueling double
	DQN model at 500 timesteps per epsiode on LunarLander-v2

	8. opt_duel_double_quarter_LL.ipynb: Jupyter notebook containing the trained optimised dueling 
	double DQN model at 250 timesteps per epsiode on LunarLander-v2

	9. opt_duel_LL.ipynb: Jupyter notebook containing the trained optimised dueling DQN model at 
	1000 timesteps per epsiode on LunarLander-v2

	10. opt_LL.ipynb: Jupyter notebook containing the trained optimised vanilla DQN model at 
	1000 timesteps per epsiode on LunarLander-v2

	11. optunaLL_multi_trial_1.3.ipynb: Jupyter notebook containing hyperparameter tuning of the 
	vanilla DQN on LunarLander-v2 on LunarLander-v2
	
	12. baseline_LL.ipynb:Jupyter notebook containing the trained random policy DQN model at 
	1000 timesteps per epsiode on LunarLander-v2

	13. vid_tools.py: Python file containing functions for recording the Q-learning agents with the reference 
	used to create the code


results:

	1. baseline: directory containing CSV and checkpoints for the baseline (random) DQN trained on LunarLander-v2

	2. double: directory containing CSV and checkpoints for the optimised double DQN trained on LunarLander-v2

	3. duel: directory containing CSV and checkpoints for the optimised dueling DQN trained on LunarLander-v2

	4. duel_dQN: directory containing CSV and checkpoints for the optimised dueling double DQN trained on LunarLander-v2

	5. opt: directory containing CSV and checkpoints for the optimised vanilla DQN trained on LunarLander-v2

	6. model_comp.csv: csv file containing aggregated training results for all DQNs trained using 1000 steps per episodes

	7. time_comp.csv: csv file containing aggregated training results for dueling double DQNS trained using varying
	max episode lengths


*******************************************************************************************************

ATARI

1. ray_results: directory containig csv files from DQN and PPO agents trained on PongDeterministic-v4 over 1.1M timesteps

2. pong_optDQN.ipynb: Jupyter notebook containign DQN agent trained on pong using RLLIB using the config suggested by RLLIB
(LARGE FILE CAN CRASH UPON OPENING)

3. pong_optDQN.ipynb: Jupyter notebook containign PPO agent trained on pong using RLLIB using the config suggested by RLLIB
(LARGE FILE CAN CRASH UPON OPENING)

4. DRL.atari.vis.ipynb: Jupyter notebook containing visualisations of training scores and episode lenghths for PPO and 
DQN agents

	

