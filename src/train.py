import math
import torch
from torch.distributions import Categorical
import numpy as np
from collections import deque

# ignore import error 
import gymnasium as gym
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

from model import MLP
from environments import SampleGymnasiumEnv

# ========== Config ==========

max_episodes = 1000
max_steps = 10000
seed = 1337
eps = np.finfo(np.float32).eps.item()

# ========== Init ===========

# toy text environments do not seem to give any observation
gymnasium_env = gym.make("CartPole-v1")
gymnasium_env = NumpyToTorch(gymnasium_env)
policy = MLP(4, 2, dropout=0.6)  
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

gymnasium_env.reset(seed=seed)
torch.manual_seed(seed)

reward_buffer = []
ap_buffer = []

# ========== Loop ===========

for episode in range(max_episodes):
    observation, info = gymnasium_env.reset()
    terminal, truncation = False, False
    reward_buffer = []
    ap_buffer = []
    for step in range(max_steps):
        # sample one action from the policy
        # apparently you cannot use torch multinomial here since it doesnt work with gradients
        ap = policy(observation).unsqueeze(0)
        m = Categorical(ap)
        action = m.sample()
        ap_buffer.append(m.log_prob(action))
        # convert to integer
        action = int(action)
        observation, reward, terminal, truncation, info = gymnasium_env.step(action)
        reward_buffer.append(reward)

        if(terminal or truncation):
            break

    R = 0
    policy_loss = []
    returns = deque()
    for r in reward_buffer[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    # normalize the returns, not strictly necessary
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # zip just yields the elements from ap_buffer and returns
    for log_prob, R in zip(ap_buffer, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    # the loss is the sum of log_prob times the discounted reward
    # for each step
    # negative because we want to maximize 
    # using the gradient
    # we sum here but this is just pytorch syntax
    # we compute the gradient for each step and add it to the network parameters
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    if(episode % 10 == 0):
        print(f"Episode: {episode}")
        print(f"Reward: {sum(reward_buffer)}")