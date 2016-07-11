import gym
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from logging import getLogger

from .memory import Memory
from .network import Network
from .utils import get_timestamp
from .exploration import OUExploration
from .statistic import Statistic

logger = getLogger(__name__)

class NAF(object):
  def __init__(self,
               sess, model_dir, env_name,
               use_batch_norm, l1_reg_scale, l2_reg_scale,
               hidden_dims, hidden_activation_fn,
               tau, decay, epsilon, discount,
               memory_size, batch_size,
               learning_rate,
               max_step, max_update, max_episode, test_step):
    self.sess = sess
    self.model_dir = model_dir
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.strategy = OUExploration(self.env)
    self.action_size = self.env.action_space.shape[0]

    assert isinstance(self.env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(self.env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    self.discount = discount
    self.max_step = max_step
    self.max_update = max_update
    self.max_episode = max_episode
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.memory = Memory(self.env, batch_size, memory_size)

    shared_args = {
      'session': sess,
      'input_shape': self.env.observation_space.shape,
      'action_size': self.env.action_space.shape[0],
      'use_batch_norm': use_batch_norm,
      'l1_reg_scale': l1_reg_scale, 'l2_reg_scale': l2_reg_scale,
      'hidden_dims': hidden_dims, 'hidden_activation_fn': hidden_activation_fn,
      'decay': decay, 'epsilon': epsilon,
    }

    logger.info("Creating prediction network...")
    self.pred_network = Network(
      name='pred_network', **shared_args
    )

    logger.info("Creating target network...")
    self.target_network = Network(
      name='target_network', **shared_args
    )
    self.target_network.make_soft_update_from(self.pred_network, tau)

    self.stat = Statistic(sess, env_name, test_step, 0, model_dir, self.pred_network.variables)

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
        # 3. perceive
        if is_train:
          q, v, a, loss, is_update = self.perceive(state, reward, action, terminal)

          if self.stat and is_update:
            self.stat.on_step(action, reward, terminal, q, v, a, loss, is_update)

        episode_reward += reward
        if terminal:
          self.strategy.reset()
          break

      logger.info('Episode reward: %s' % episode_reward)

    if monitor:
      self.env.monitor.close()

  def predict(self, state):
    u = self.pred_network.predict([state])[0]
    action = self.strategy.add_noise(u)

    return action

  def perceive(self, state, reward, action, terminal):
    self.memory.add(state, reward, action, terminal)

    if self.memory.count > self.batch_size:
      is_update = True
      q, v, a, loss = self.q_learning_minibatch()
    else:
      is_update, q, v, a, loss = False, None, None, None, None

    return q, v, a, loss, is_update

  def q_learning_minibatch(self):
    losses = []

    for iteration in xrange(self.max_update):
      x_t, u_t, r_t, x_t_plus_1, terminal = self.memory.sample()

      v = self.target_network.V.eval({
        self.target_network.x: x_t_plus_1,
        self.target_network.u: u_t,
        self.target_network.is_training: False,
      })

      target_v = self.discount * np.squeeze(v) + r_t
      _, q_t, v_t, a_t, loss = self.sess.run([
          self.optim,
          self.pred_network.Q,
          self.pred_network.V,
          self.pred_network.A,
          self.pred_network.loss
        ], {
          self.pred_network.target_y: target_v,
          self.pred_network.x: x_t,
          self.pred_network.u: u_t,
          self.pred_network.is_training: True,
        })
      losses.append(loss)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("target_v: %s, q_t: %s, v_t: %s, a_t: %s, loss: %s" % (target_v[0], q_t[0], v_t[0], a_t[0], loss))

    return q_t, v_t, a_t, np.mean(losses)
