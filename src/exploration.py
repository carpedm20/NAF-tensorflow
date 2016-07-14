import numpy as np
import numpy.random as nr

class Exploration(object):
  def __init__(self, env):
    self.action_size = env.action_space.shape[0]

  def add_noise(self, action, info={}):
    pass

  def reset(self):
    pass

class OUExploration(Exploration):
  # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py

  def __init__(self, env, sigma=0.3, mu=0, theta=0.15):
    super(OUExploration, self).__init__(env)

    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.state = np.ones(self.action_size) * self.mu
    self.reset()

  def add_noise(self, action, info={}):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
    self.state = x + dx

    return action + self.state

  def reset(self):
    self.state = np.ones(self.action_size) * self.mu

class LinearDecayExploration(Exploration):
  def __init__(self, env):
    super(LinearDecayExploration, self).__init__(env)

  def add_noise(self, action, info={}):
    return action + np.random.randn(self.action_size) / (info['idx_episode'] + 1)

class BrownianExploration(Exploration):
  def __init__(self, env, noise_scale):
    super(BrownianExploration, self).__init__(env)

    raise Exception('not implemented yet')
