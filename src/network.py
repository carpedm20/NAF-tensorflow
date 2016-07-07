import tensorflow as tf

from .ops import fully_connected, initializers

class Network(object):
  def __init__(self,
               sess,
               state_shape,
               action_size,
               hidden_dims,
               use_batch_norm=True,
               activation_fn=tf.nn.relu,
               weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.0),
               name='NAF'):
    self.sess = sess

    with tf.variable_scope(name):
      x = hidden_layer = tf.placeholder(tf.float32, [None] + list(state_shape), name='state')
      u = tf.placeholder(tf.float32, [None, action_size], name='action')

      variables = []

      hidden_layer = self.state
      for idx, hidden_dim in enumerate(hidden_dims):
        hidden_layer = fully_connected(
          hidden_layer,
          num_outputs=hidden_dim,
          activation_fn=activation_fn,
          weights_initializer=weights_initializer,
          biases_initializer=biases_initializer,
          variables_collections=variables,
          scope='hid%d' % idx,
        )

        if use_batch_norm:
          mean, variance = tf.nn.moments(
              hidden_layer, [0], keep_dims=True, name='moments%d' % idx)
          hidden_layer = tf.nn.batch_normalization(
              hidden_layer, mean, variance, name='batch_norm%d' % idx)

      V = fully_connected(
        hidden_layer,
        num_outputs=action_size,
        activation_fn=None,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
        variables_collections=variables,
        scope='V',
      )

      l = fully_connected(
        hidden_layer,
        num_outputs=(action_size * (action_size + 1))/2,
        activation_fn=None,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
        variables_collections=variables,
        scope='L',
      )

      columns = []
      for idx in xrange(action_size):
        column = tf.pad(tf.slice(l, (0, 0), (-1, action_size - idx)), ((0, 0), (idx, 0)))
        columns.append(column)
      L = tf.concat(1, columns)
      P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))

      mu = fully_connected(
        hidden_layer,
        num_outputs=action_size,
        activation_fn=None,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
        variables_collections=variables,
        scope='mu',
      )

      tmp = u - mu
      A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(P, tmp))/2
      Q = A + V

      with tf.variable_scope('optimizer'):
        true_action = tf.placeholder(tf.int64, [None], name='action')

        action_one_hot = tf.one_hot(true_action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
        acted_Q = tf.reduce_sum(Q * action_one_hot, reduction_indices=1, name='q_acted')

        target_Q = tf.placeholder(tf.float32, [None], name='target_Q')
        loss = tf.reduce_mean(tf.square(target_Q - acted_Q), name='loss')

      self.input = x
      self.action = u
      self.true_action = true_action
      self.target_Q = target_Q

      self.V, self.L, self.P, self.mu, self.A, self.Q = V, L, P, mu, A, Q

      self.loss = loss
      self.variables = variables

  def predict(self, state):
    return self.sess.run(self.mu, {self.input: state})

  def make_copy_from(self, network):
    self.assign_op = {}

    for from_, to_ in zip(network.variables, self.variables):
      self.assign_op[self.to_.name] = to_.assign(from_)

  def update_from(self, network):
    for variable in self.variables:
      self.sess.run(self.assign_op[variable.name])
