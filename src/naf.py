from logging import getLogger
logger = getLogger(__name__)

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables

from .utils import get_timestamp

class NAF(object):
  def __init__(self, sess,
               env, strategy, pred_network, target_network, stat,
               discount, batch_size, learning_rate,
               max_steps, update_repeat, max_episodes):
    self.sess = sess
    self.env = env
    self.strategy = strategy
    self.pred_network = pred_network
    self.target_network = target_network
    self.stat = stat

    self.discount = discount
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.action_size = env.action_space.shape[0]

    self.max_steps = max_steps
    self.update_repeat = update_repeat
    self.max_episodes = max_episodes

    self.prestates = []
    self.actions = []
    self.rewards = []
    self.poststates = []
    self.terminals = []

    with tf.name_scope('optimizer'):
      self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
      self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(self.pred_network.Q)), name='loss')

      self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  def run(self, monitor=False, display=False, is_train=True):
    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

    if monitor:
      self.env.monitor.start('/tmp/%s-%s' % (self.stat.env_name, get_timestamp()))

    for self.idx_episode in xrange(self.max_episodes):
      state = self.env.reset()

      for t in xrange(0, self.max_steps):
        if display: self.env.render()

        # 1. predict
        action = self.predict(state)

        # 2. step
        self.prestates.append(state)
        state, reward, terminal, _ = self.env.step(action)
        self.poststates.append(state)

        terminal = True if t == self.max_steps - 1 else terminal

        # 3. perceive
        if is_train:
          q, v, a, l = self.perceive(state, reward, action, terminal)

          if self.stat:
            self.stat.on_step(action, reward, terminal, q, v, a, l)

        if terminal:
          self.strategy.reset()
          break

    if monitor:
      self.env.monitor.close()

  def run2(self, monitor=False, display=False, is_train=True):
    target_y = tf.placeholder(tf.float32, [None], name='target_y')
    loss = tf.reduce_mean(tf.squared_difference(target_y, tf.squeeze(self.pred_network.Q)), name='loss')

    optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

    # replay memory
    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []

    # the main learning loop
    total_reward = 0
    for i_episode in xrange(self.max_episodes):
      observation = self.env.reset()
      episode_reward = 0

      for t in xrange(self.max_steps):
        if display:
          self.env.render()

        # predict the mean action from current observation
        x_ = np.array([observation])
        u_ = self.pred_network.mu.eval({self.pred_network.x: x_})[0]

        action = u_ + np.random.randn(1) / (i_episode + 1)

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = self.env.step(action)
        episode_reward += reward

        rewards.append(reward); poststates.append(observation); terminals.append(done)

        if len(prestates) > 10:
          loss_ = 0
          for k in xrange(self.update_repeat):
            if len(prestates) > self.batch_size:
              indexes = np.random.choice(len(prestates), size=self.batch_size)
            else:
              indexes = range(len(prestates))

            # Q-update
            v_ = self.target_network.V.eval({self.target_network.x: np.array(poststates)[indexes]})
            y_ = np.array(rewards)[indexes] + self.discount * np.squeeze(v_)

            tmp1, tmp2 = np.array(prestates)[indexes], np.array(actions)[indexes]
            loss_ += l_

            self.target_network.soft_update_from(self.pred_network)

        if done:
          break

      print "average loss:", loss_/k
      print "Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward)
      total_reward += episode_reward

    print "Average reward per episode {}".format(total_reward / self.episodes)

  def predict(self, state):
    u = self.pred_network.predict([state])[0]

    return self.strategy.add_noise(u, {'idx_episode': self.idx_episode})

  def perceive(self, state, reward, action, terminal):
    self.rewards.append(reward)
    self.actions.append(action)

    return self.q_learning_minibatch()

  def q_learning_minibatch(self):
    q_list = []
    v_list = []
    a_list = []
    l_list = []

    for iteration in xrange(self.update_repeat):
      if len(self.rewards) >= self.batch_size:
        indexes = np.random.choice(len(self.rewards), size=self.batch_size)
      else:
        indexes = np.arange(len(self.rewards))

      x_t = np.array(self.prestates)[indexes]
      x_t_plus_1 = np.array(self.poststates)[indexes]
      r_t = np.array(self.rewards)[indexes]
      u_t = np.array(self.actions)[indexes]

      v = self.target_network.predict_v(x_t_plus_1, u_t)
      target_y = self.discount * np.squeeze(v) + r_t

      _, l, q, v, a = self.sess.run([
        self.optim, self.loss,
        self.pred_network.Q, self.pred_network.V, self.pred_network.A,
      ], {
        self.target_y: target_y,
        self.pred_network.x: x_t,
        self.pred_network.u: u_t,
        self.pred_network.is_train: True,
      })

      q_list.extend(q)
      v_list.extend(v)
      a_list.extend(a)
      l_list.append(l)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("q: %s, v: %s, a: %s, l: %s" \
        % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))

    return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)
