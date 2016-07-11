import tensorflow as tf
from logging import getLogger

from .ops import fully_connected, initializers, get_variables, batch_norm, l1_regularizer, l2_regularizer

logger = getLogger(__name__)
He_uniform = initializers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

class Network(object):
  def __init__(self,
               session,
               input_shape,
               action_size,
               hidden_dims,
               decay=0.9,
               epsilon=1e-4,
               use_batch_norm=False,
               l1_reg_scale=0.001,
               l2_reg_scale=None,
               action_merge_layer=-2,
               hidden_activation_fn=tf.nn.tanh,
               hidden_weights_initializer=He_uniform,
               hidden_biases_initializer=tf.constant_initializer(0.0),
               output_activation_fn=None,
               output_weights_initializer=tf.random_uniform_initializer(-3e-4,3e-4),
               output_biases_initializer=tf.constant_initializer(0.0),
               name='NAF'):
    self.sess = session

    if l1_reg_scale != None:
      regularizer = l1_regularizer(l1_reg_scale)
    elif l2_reg_scale != None:
      regularizer = l2_regularizer(l2_reg_scale)
    else:
      regularizer = None

    with tf.variable_scope(name):
      x = tf.placeholder(tf.float32, (None,) + tuple(input_shape), name='observations')
      u = tf.placeholder(tf.float32, (None, action_size), name='actions')

      is_training = tf.placeholder(tf.bool, name='is_training')

      n_layers = len(hidden_dims) + 1
      if n_layers > 1:
        action_merge_layer = \
          (action_merge_layer % n_layers + n_layers) % n_layers
      else:
        action_merge_layer = 1

      logger.debug("hidden_dims: %s" % hidden_dims)
      logger.debug("action_merge_layer: %d" % action_merge_layer)

      if use_batch_norm:
        hidden_layer = batch_norm(x, decay=decay, epsilon=epsilon, is_training=is_training)
      else:
        hidden_layer = x

      for idx, hidden_dim in enumerate(hidden_dims):
        if use_batch_norm:
          batch_norm_args = {
            'normalizer_fn': batch_norm,
            'normalizer_params': {'decay': decay, 'epsilon': epsilon, 'is_training': is_training}
          }
        else:
          batch_norm_args = {}
                
        hidden_layer = fully_connected(
          hidden_layer,
          num_outputs=hidden_dim,
          activation_fn=hidden_activation_fn,
          weights_initializer=hidden_weights_initializer,
          weights_regularizer=regularizer,
          biases_initializer=hidden_biases_initializer,
          scope='hid%d' % idx,
          **batch_norm_args
        )

      def make_output(layer, num_outputs, scope='out'):
        return fully_connected(
          layer,
          num_outputs=num_outputs,
          activation_fn=output_activation_fn,
          weights_initializer=output_weights_initializer,
          weights_regularizer=regularizer,
          biases_initializer=output_biases_initializer,
          scope=scope,
        )

      with tf.variable_scope('value'):
        V = make_output(hidden_layer, 1, scope='V')

      with tf.variable_scope('advantage'):
        l = make_output(hidden_layer, (action_size * (action_size + 1))/2, scope='l')
        mu = make_output(hidden_layer, action_size, scope='mu')

        pivot = 0
        rows = []
        for idx in xrange(action_size):
          count = action_size - idx

          diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
          non_diag_elems = tf.slice(l, (0, pivot+1), (-1, count-1))
          row = tf.pad(tf.concat(1, (diag_elem, non_diag_elems)), ((0, 0), (idx, 0)))
          rows.append(row)

          pivot += count

        L = tf.transpose(tf.pack(rows, axis=1), (0, 2, 1))
        P = tf.batch_matmul(L, tf.transpose(L, (0, 2, 1)))

        tmp = tf.expand_dims(u - mu, -1)
        A = -tf.batch_matmul(tf.transpose(tmp, [0, 2, 1]), tf.batch_matmul(P, tmp))/2
        A = tf.reshape(A, [-1, 1])

      Q = A + V

      with tf.variable_scope('optimizer'):
        target_y = tf.placeholder(tf.float32, [None], name='target_y')
        loss = tf.reduce_mean(tf.square(target_y - Q), name='loss')

      self.x = x
      self.u = u
      self.loss = loss
      self.target_y = target_y
      self.is_training = is_training

      self.V, self.l, self.L, self.P, self.mu, self.A, self.Q = V, l, L, P, mu, A, Q
      self.variables = get_variables(name)

  def predict(self, state):
    return self.sess.run(self.mu, {self.x: state, self.is_training: False})

  def make_soft_update_from(self, network, tau):
    logger.info("Creating ops for soft target update...")
    assert len(network.variables) == len(self.variables), \
      "target and prediction network should have same # of variables"

    self.assign_op = {}
    for from_, to_ in zip(network.variables, self.variables):
      if 'BatchNorm' in to_.name:
        print "assign", to_.name
        self.assign_op[to_.name] = to_.assign(from_)
      else:
        print "soft", to_.name
        self.assign_op[to_.name] = to_.assign(tau * from_ + (1-tau) * to_)

  def hard_copy_from(self, network):
    logger.info("Creating ops for hard target update...")
    assert len(network.variables) == len(self.variables), \
      "target and prediction network should have same # of variables"

    for from_, to_ in zip(network.variables, self.variables):
      self.sess.run(to_.assign(from_))

  def soft_update_from(self, network):
    for variable in self.variables:
      self.sess.run(self.assign_op[variable.name])
    return True
