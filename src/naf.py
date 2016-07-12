import numpy as np
from tqdm import tqdm
import tensorflow as tf
from logging import getLogger

from .utils import get_timestamp

logger = getLogger(__name__)

class NAF(object):
  def __init__(self,
               env, strategy, pred_network, target_network, stat,
               discount, batch_size, learning_rate,
               max_step, max_update, max_episode):
    self.env = env
    self.strategy = strategy
    self.pred_network = pred_network
    self.target_network = target_network
    self.stat = stat

    self.discount = discount
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.max_step = max_step
    self.max_update = max_update
    self.max_episode = max_episode

    self.states = []
    self.rewards = []
    self.actions = []

  def run(self, monitor=False, display=False, is_train=True):
    self.optim = tf.train.AdamOptimizer(self.learning_rate) \
      .minimize(self.pred_network.loss, var_list=self.pred_network.variables)

    self.stat.load_model()

    if monitor:
      self.env.monitor.start('/tmp/%s-%s' % (self.env_name, get_timestamp()))

    self.target_network.hard_copy_from(self.pred_network)
    for self.idx_episode in tqdm(range(0, self.max_episode), ncols=70):
      episode_reward = 0
      state = self.env.reset()

      for t in xrange(0, self.max_step):
        if display: self.env.render()

        # 1. predict
        action = self.predict(state)

        # 2. step
        state, reward, terminal, _ = self.env.step(action)
        terminal = True if t == self.max_step-1 else terminal

        # 3. perceive
        if is_train:
          q, v, a, l = self.perceive(state, reward, action, terminal)

          if self.stat:
            self.stat.on_step(action, reward, terminal, q, v, a, l)

        episode_reward += reward
        if terminal:
          self.strategy.reset()
          break

    if monitor:
      self.env.monitor.close()

  def predict(self, state):
    u = self.pred_network.predict([state])[0]
    action = self.strategy.add_noise(u)
    return action

  def perceive(self, state, reward, action, terminal):
    self.states.append(state)
    self.rewards.append(reward)
    self.actions.append(action)

    return self.q_learning_minibatch()

  def q_learning_minibatch(self):
    q_list = []
    v_list = []
    a_list = []
    l_list = []

    for iteration in xrange(self.max_update):
      if len(self.states) - 1 > self.batch_size:
        indexes = np.random.choice(len(self.states) - 1, size=self.batch_size)
      else:
        indexes = np.arange(len(self.states) - 1)

      x_t = np.array(self.states)[indexes]
      x_t_plus_1 = np.array(self.states)[indexes + 1]
      r_t = np.array(self.rewards)[indexes]
      u_t = np.array(self.actions)[indexes]

      v = self.target_network.V.eval({
        self.target_network.x: x_t_plus_1,
        self.target_network.u: u_t,
        self.target_network.is_training: False,
      })

      target_v = self.discount * np.squeeze(v) + r_t
      q, v, a, l = self.pred_network.update(self.optim, target_v, x_t, u_t)

      q_list.extend(q)
      v_list.extend(v)
      a_list.extend(a)
      l_list.append(l)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("q: %s, v: %s, a: %s, l: %s" \
        % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))

    return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)
