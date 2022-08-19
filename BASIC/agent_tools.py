import random
import gym
import numpy as np
import time
from IPython.display import clear_output

class Q_learner:
    def __init__(self, env, state_space, action_space, alpha, gamma, epsilon, decay_type, decay_steps, decay_end, random_policy):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_space = state_space
        self.action_space = action_space
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.decay_end = decay_end
        self.random_policy = random_policy
  

    def Qtable(self, state_space, action_space, bin_size):
        """Create Q-table from discrete space"""
        bins = [np.linspace(-4.8,4.8,bin_size),
                np.linspace(-4,4,bin_size),
                np.linspace(-0.418,0.418,bin_size),
                np.linspace(-4,4,bin_size)]

        q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
        return q_table, bins
    
    def Discrete(self, state, bins):
        index = []
        for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
        return tuple(index)

    def decide_action(self,state):
        #intialise environment
        current_state = self.Discrete(state, self.bins)
        action = np.argmax(self.q_table[current_state]) #exploit
        return action

    def train(self, episodes):
        #create performance list and q-table
        scores_list, master_list = [], []
        self.q_table, self.bins = self.Qtable(self.state_space, self.action_space, 30)
        
        #for decay function
        alpha  = self.alpha
        epsilon = self.epsilon
        gamma = self.gamma
        decay_end = self.decay_end
        decay_steps = self.decay_steps
        decay_type = self.decay_type
        
        if decay_type == 'epsilon':
            epsilon_diff = epsilon - decay_end
            decay_step = epsilon_diff/decay_steps
                    
        
        #create training loop
        for episode in range(1, episodes+1):
            
            #create initial time
            score = 0
            
            #intialise environment
            current_state = self.Discrete(self.env.reset(),self.bins)
            done = False #instantiate game loop

            #start q-learning loop
            while not done:
                
                if self.random_policy == True:
                    action = self.env.action_space.sample() #explore
                    
                elif random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() #explore
                    
                else:
                    action = np.argmax(self.q_table[current_state]) #exploit

                #update Q-table
                observation, reward, done, info = self.env.step(action) 
                next_state = self.Discrete(observation,self.bins)
                score += reward
                
                #update q-table
                if not done:
                    max_future_q = np.max(self.q_table[next_state])
                    current_q = self.q_table[current_state+(action,)]
                    new_q = (1-alpha)*current_q + alpha*(reward + self.gamma*max_future_q)
                    self.q_table[current_state+(action,)] = new_q
                    

                                    
                #save the scores
                current_state = next_state   
            
            #get scores
            episode_score = score/500 #200 is max number of steps for cartpole v0 and 500 for cartpole v1
            scores_list.append(episode_score)
            
            #update decay parameters
            if decay_type == 'epsilon':
                epsilon  = epsilon - decay_step
                #print(epsilon)
                
            #append scores every 100 steps (solved length)
            if episode % 100 == 0:
                master_list.append(scores_list)
                scores_list = []
                
            #show help    
            if episode % 100 == 0:
                clear_output(wait=True)#
                print(f"Episode: {episode}")
                
                      
        #calculate and return objective function
        return master_list

    
def performance_splitter(results):
    per75 = [np.percentile(x, 75) for x in results]
    per50 = [np.percentile(x, 50) for x in results]
    per25 = [np.percentile(x, 25) for x in results]
    return [per75, per50, per25]
