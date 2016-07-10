import gym
from tqdm import tqdm
import tensorflow as tf

from .network import Network
from .memory import Memory
from .utils import get_timestamp

class NAF(object):
  def __init__(self,
               sess,
               env_name,
               memory_size,
               batch_size,
               discount):
    self.env_name = env_name
    self.env = gym.make(env_name)

    assert isinstance(self.env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(self.env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    self.discount = discount
    self.memory = Memory(self.env, batch_size, memory_size)

    self.pred_network = Network(
      session=sess,
      input_shape=self.env.observation_space.shape,
      action_size=self.env.action_space.shape[0],
      hidden_dims=[200, 200],
      name='pred_network',
    )
    self.target_network = Network(
      session=sess,
      input_shape=self.env.observation_space.shape,
      action_size=self.env.action_space.shape[0],
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
    self.optim = tf.train.AdamOptimizer(learning_rate) \
      .minimize(self.pred_network.loss, global_step=step_op)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver(self.pred_network.variables + [step_op], max_to_keep=30)

    if monitor:
      self.env.monitor.start('/tmp/%s-%s' % (self.env_name, get_timestamp()))

    start_step = step_op.eval()
    iterator = tqdm(range(start_step, num_train), ncols=70, initial=start_step)

    for self.step in iterator:
      state = self.env.reset()

      for t in xrange(max_step):
        if display: self.env.render()

        action = self.pred_network.predict([state])[0]

        state, reward, terminal, _ = self.env.step(action)
        self.memory.add(state, reward, action, terminal)

        if self.memory.size > learn_start:
          self.q_learning_minibatch()

        if self.step % self.target_q_update_step == self.target_q_update_step - 1:
          self.update_target_q_network()

        if terminal: break

    if mointor:
      self.env.monitor.close()

  def predict(self, s_t, test_ep=None):
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def q_learning_minibatch(self):
    total_loss = 0

    for iteration in xrange(max_update):
      x_t, u_t, r_t, x_t_plus_1, terminal = self.memory.sample()

      q_t_plus_1 = self.target_network.q.eval({self.x: x_t, self.u: u_t})

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + r_t

      _, q_t, loss = self.sess.run(
        [self.optim, self.pred_network.q, self.pred_entwork.loss], {
          self.target_Q: target_q_t,
          self.pred_network.x: x_t,
          self.pred_network.u: u_t,
        })

      total_loss += loss

    return total_loss
