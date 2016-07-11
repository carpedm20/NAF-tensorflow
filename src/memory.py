import os
import random
import logging
import numpy as np

class Memory:
  def __init__(self, env, batch_size, memory_size, data_format='NCHW'):
    self.data_format = data_format
    self.memory_size = memory_size
    self.action_shape = env.action_space.shape
    self.observation_shape = env.observation_space.shape

    self.rewards = np.empty(self.memory_size, dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.actions = np.empty((self.memory_size,) + self.action_shape, dtype = np.float16)
    self.screens = np.empty((self.memory_size,) + self.observation_shape, dtype = np.float16)

    self.count = 0
    self.current = 0
    self.batch_size = batch_size

    self.prestates = np.empty(
        (self.batch_size,) + self.observation_shape, dtype = np.float16)
    self.poststates = np.empty(
        (self.batch_size,) + self.observation_shape, dtype = np.float16)
    self.batch_actions = np.empty(
        (self.batch_size,) + self.action_shape, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    self.rewards[self.current] = reward
    self.terminals[self.current] = terminal
    self.actions[self.current, ...] = action
    self.screens[self.current, ...] = screen

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def get_state_from(self, array, index):
    index = index % self.count
    return array[index:index + 1, ...]

  def sample(self, count=None):
    indexes = []

    if count == None:
      count = self.batch_size

    while len(indexes) < count:
      while True:
        index = random.randint(0, self.count - 1)
        if index >= self.current:
          continue
        break
      
      self.prestates[len(indexes), ...] = self.get_state_from(self.screens, index - 1)
      self.poststates[len(indexes), ...] = self.get_state_from(self.screens, index)
      self.batch_actions[len(indexes), ...] = self.get_state_from(self.actions, index - 1)

      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.data_format == 'NHWC' and len(self.screens.shape) == 3:
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals
