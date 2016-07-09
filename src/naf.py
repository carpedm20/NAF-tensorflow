import gym
import tensorflow as tf

from .network import Network
from .memory import Memory
from .utils import get_timestamp

class NAF(object):
  def __init__(self, env_name, sess):
    self.env_name = env_name
    self.env = gym.make(env_name)
    assert isinstance(self.env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(self.env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    self.memory = Memory()

    self.pred_network = Network(
      session=sess,
      input_shape=self.env.observation_space.shape,
      action_size=self.env.action_space.shape[0] + 1,
      hidden_dims=[200, 200],
      name='pred_network',
    )
    self.target_network = Network(
      session=sess,
      input_shape=self.env.observation_space.shape,
      action_size=self.env.action_space.shape[0] + 1,
      hidden_dims=[200, 200],
      name='target_network',
    )

    self.target_network.make_copy_from(self.pred_network)

  def train(self,
            num_train,
            learning_rate,
            learn_start,
            max_step,
            max_update,
            monitor,
            display=False):
    step_op = tf.Variable(0, trainable=False, name='step')
    optim = tf.train.AdamOptimizer(learning_rate) \
      .minimize(self.pred_network.loss, global_step=step_op)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver(self.pred_network.variables + [step_op], max_to_keep=30)

    if monitor:
      self.env.monitor.start('/tmp/%s-%s' % (self.env_name, get_timestamp()))
    for episode in xrange(num_train):
      state = self.env.reset()

      for t in xrange(max_step):
        if display: self.env.render()

        action = self.pred_network.predict([state])[0]

        state_, reward, terminal, _ = self.env.step(action)
        self.memory.add(state, action, reward, terminal, state_)
        state = state_

        if self.memory.size > learn_start:
          loss = 0

          for iteration in xrange(max_update):
            prestates, actions, rewards, terminals, poststates = self.memory.sample()

        if terminal: break

    if mointor:
      self.env.monitor.close()
