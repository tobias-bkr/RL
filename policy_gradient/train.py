import yaml

import sys
import os

import math
import time
from collections import deque

import numpy as np

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.experimental.wrappers.numpy_to_torch import NumpyToTorchV0

import pufferlib.emulation
from pufferlib.ocean.breakout.breakout import Breakout

from model import prob_MLP
from model import MLP
from model import smolMLP
from environments import SampleGymnasiumEnv

# ========== Functions ===========

def linear_anneal(c, episode):
    return c["lr"] - (episode - 1) * (c["lr"] / c["max_episodes"])

def choose_action(policy, observation, ap_buffer):
    # sample one action from the policy
    # apparently you cannot use torch multinomial here since it doesn't really work with gradients
    ap = policy(observation).unsqueeze(0)
    m = Categorical(ap)
    action = m.sample()
    ap_buffer.append(m.log_prob(action))
    # convert to integer
    return int(action)

def GAE():
    return 0

class run_baseline():
    def __init__(self):
        self.sum = 0
        self.samples = 0
    def calc_baseline(self, returns):
        """expects 1D list/array"""
        self.samples += len(returns)
        self.sum += sum(returns)
        return self.sum / self.samples
    def __call__(self, returns, observations):
        return self.calc_baseline(returns)
    
class episode_baseline():
    def calc_baseline(self, returns):
        """expects 1D pytorch tensor"""
        return returns.mean()
    def __call__(self, returns, observations):
        return self.calc_baseline(returns)

# TODO implement
class critic_baseline():
    def __init__(self):
        return
    def calc_baseline(self, returns, observations):
        # do MSE loss on collected returns
        # output predictions for each observation
        return
    def __call__(self, returns, observations):
        return self.calc_baseline(returns, observations)

def REINFORCE(optimizer, ap_buffer, reward_buffer, observation_buffer, discount_factor, eps):
    discounted_reward = 0
    policy_loss = []
    returns = deque()
    # go backwards through reward buffer
    # accumulate discounted reward for every step
    # each step is discounted by discount_factor^x where x is the distance to the current step, with x >= 0
    for reward in reward_buffer[::-1]:
        discounted_reward = reward + discount_factor * discounted_reward
        returns.appendleft(discounted_reward)
    returns = torch.tensor(returns)
    # Z normalize the returns, not strictly necessary
    # works kind of like a baseline
    returns = (returns - baseline(returns, observation_buffer)) / (returns.std() + eps)
    # zip just yields the elements from ap_buffer and returns
    for log_prob, discounted_reward in zip(ap_buffer, returns):
        policy_loss.append(-log_prob * discounted_reward)
    # the loss is the sum of log_prob times the discounted reward
    # for each step
    # negative because we want to maximize 
    # using gradient descent
    # we compute the gradient for each step and subtract it from the network parameters
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    # do not accumulate gradients 
    optimizer.zero_grad(set_to_none = True)
    return

# ========== Config ==========

config_path = "./policy_gradient/configs/Cartpole.yaml"
state_path = "./policy_gradient/checkpoints/"
eval = False

# ========== Init ==========

# open config as c for readability
with open(config_path, "r") as cf:
    c = yaml.safe_load(cf)

# config correction if eval
if(eval):
    c["max_episodes"] = 1
    c["max_steps_per_episode"] = sys.maxsize # basically infinite
    c["terminal_log_interval"] = 1 # print results of the episode
    c["tensorboard_log_interval"] = 2 # do not log in tensorboard
    c["save_interval"] = 2 # do not save
    c["render_mode"] = "human"
else:
    writer = SummaryWriter(log_dir=f"./policy_gradient/runs/{c['model_name']}/") 
    os.makedirs(f"./policy_gradient/checkpoints/{c['model_name']}", exist_ok=True)

# only one checkpoint per model for now
model_path = f"./policy_gradient/checkpoints/{c['model_name']}/{c['model_name']}_model1.pt"
optimizer_path = f"./policy_gradient/checkpoints/{c['model_name']}/{c['model_name']}_optimizer1.pt"
state_path = f"./policy_gradient/checkpoints/{c['model_name']}/{c['model_name']}_state1.yaml" 

# init baseline
match(c["baseline"]):
    case "episode_baseline":
        baseline = episode_baseline()
    case "run_baseline":
        baseline = run_baseline()
    case "critic_baseline":
        baseline = critic_baseline()
    case _:
        raise ValueError("unknown baseline")

# init policy
match(c["policy"]):
    case "prob_MLP":
        policy = prob_MLP(c["d_i"], c["d_o"], dropout=c["dropout"])
    case "smolMLP":
        policy = smolMLP(c["d_i"], c["d_o"], dropout=c["dropout"])
    case _:
        raise ValueError("unknown policy")
try:
    policy.load_state_dict(torch.load(model_path, weights_only=True))
except FileNotFoundError:
    print("Model does not exist, a new one will be created")
policy.train()  

# init optimizer
match(c["optimizer"]):
    case "SGD":
        optimizer = torch.optim.SGD(policy.parameters(), lr=c["lr"])
    case "Adam":
        optimizer = torch.optim.Adam(policy.parameters(), lr=c["lr"], betas=((0.9,0.999)))
    case _:
        raise ValueError("unknown optimizer")
try:
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
except FileNotFoundError:
    print("Optimizer does not exist, a new one will be created")
    
# init env
match(c["env_type"]):
    case "Gymnasium":
        # toy text environments do not seem to give any observation unless wrapping with tensors
        env = gym.make(c["env_name"], render_mode=c["render_mode"])
    case "Puffer":
        pass
    case _:
        raise ValueError("unknown env type")

# still not deterministic?
torch.manual_seed(c["seed"])

# ========== Loop ==========

print("="*50)
# add already trained time later
# and how many runs it was
start_time = time.perf_counter() 

# 1-indexing
for episode in range(1, c["max_episodes"] + 1):
    observation, info = env.reset(seed=c["seed"])
    terminal, truncation = False, False
    reward_buffer = []
    ap_buffer = []
    observation_buffer = torch.empty((0,c["d_i"]))

    if(c["annealing"] is True):
        # linearly anneal learning rate from lr to lr / max_episodes
        optimizer.param_groups[0]['lr'] = linear_anneal(c, episode)

    if(c["render_mode"] is not None):
        env.render()

    # 1-indexing
    for step in range(1, c["max_steps_per_episode"] + 1):
        observation = torch.from_numpy(np.expand_dims(observation, axis=0))
        observation_buffer = torch.cat((observation_buffer, observation), dim=0)
        # choose action and save action probabilities
        action = choose_action(policy, observation, ap_buffer)
        observation, reward, terminal, truncation, info = env.step(action)
        # behaves weirdly if not casted to int
        reward_buffer.append(int(reward))

        if(c["render_mode"] is not None):
            env.render()

        if(terminal or truncation):
            break

    if(not eval):
       REINFORCE(optimizer, ap_buffer, reward_buffer, observation_buffer, c["discount_factor"], c["eps"])

    if(episode % c["terminal_log_interval"] == 0):
        print(f"Episode: {episode}")
        print(f"Reward: {sum(reward_buffer)}")
        print(f"Learning Rate:{optimizer.param_groups[0]['lr']}")
        print(f"Time elapsed: {time.perf_counter()-start_time}")
        print("="*50)

    if(episode % c["tensorboard_log_interval"] == 0):
        # first is name of section in tensorboard and both is name of graph
        # writer.add_scalar("loss/episode", loss, episode)
        writer.add_scalar("reward/episode", sum(reward_buffer), episode)
        writer.add_scalar("steps/episode", step, episode)
        writer.add_scalar("lr/episode", optimizer.param_groups[0]['lr'], episode)

        # TODO add replay buffer logging 
        writer.flush()

    if(episode % c["save_interval"] == 0):
        torch.save(policy.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)