# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py

import numpy as np
import numpy.random as nr

class OUExploration:
  def __init__(self, env, mu=0, theta=0.15, sigma=0.3):
    self.action_size = env.action_space.shape[0]
    self.lb, self.ub = env.action_space.low, env.action_space.high

    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.state = np.ones(self.action_size) * self.mu
    self.reset()

  def reset(self):
    self.state = np.ones(self.action_size) * self.mu

  def add_noise(self, action):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
    self.state = x + dx
    return np.clip(action + self.state, self.lb, self.ub)
