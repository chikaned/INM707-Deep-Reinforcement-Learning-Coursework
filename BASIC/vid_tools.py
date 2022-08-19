import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob

#REFERENCE: https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

def play_vid(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay loop controls style="height: 400px;">
                                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                    </video>'''.format(encoded.decode('ascii'))))
    else:
        print("No video avaiable!")
        
def rec_vid(agent, env_name, checkpoint_path=None, dir_path=None):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path= dir_path+"_{}.mp4".format(env_name))
    if checkpoint_path:
        agent.policy_net.load_state_dict(torch.load(checkpoint_path))
        agent.target_net.load_state_dict(torch.load(checkpoint_path))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        action = agent.decide_action(state)
        state, reward, done, _ = env.step(action)        
    env.close()