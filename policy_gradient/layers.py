import torch
from model import smolMLP

import gymnasium as gym

env = gym.make("CartPole")
obs, inf = env.reset(seed=1)
print(obs)

policy = smolMLP(4,2)
try:
    policy.load_state_dict(torch.load(("./policy_gradient/checkpoints/Cartpole-v0.0"), weights_only=True))
except FileNotFoundError:
    print("Model does not exist, a new one will be created")

for pn, p in policy.named_parameters():
    print(pn, p)
    print(p.size())

print(policy.layers(torch.from_numpy(obs)))