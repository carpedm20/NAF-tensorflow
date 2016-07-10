import tensorflow as tf

from .ops import fully_connected, initializers, get_variables, batch_norm

He_uniform = initializers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

class Network(object):
  def __init__(self,
               session,
               input_shape,
               action_size,
               hidden_dims,
               use_batch_norm=True,
               action_merge_layer=-2,
               hidden_activation_fn=tf.nn.relu,
               hidden_weights_initializer=He_uniform,
               hidden_biases_initializer=tf.constant_initializer(0.0),
               output_activation_fn=None,
               output_weights_initializer=tf.random_uniform_initializer(-3e-4,3e-4),
               output_biases_initializer=tf.constant_initializer(0.0),
               name='NAF'):
    self.sess = session

    with tf.variable_scope(name):
      x = hidden_layer = tf.placeholder(tf.float32, [None] + list(input_shape), name='observations')
      u = tf.placeholder(tf.float32, [None, action_size], name='actions')
      is_train = tf.placeholder(tf.bool, name='is_train')

      n_layers = len(hidden_dims) + 1
      if n_layers > 1:
        action_merge_layer = \
          (action_merge_layer % n_layers + n_layers) % n_layers
      else:
        action_merge_layer = 1

      hidden_layer = batch_norm(x, is_train)

      for idx, hidden_dim in enumerate(hidden_dims):
        # if batch_norm is used, apply activation_fn after batch norm,
        # and remove biases which is redundant
        hidden_layer = fully_connected(
          hidden_layer,
          num_outputs=hidden_dim,
          activation_fn=hidden_activation_fn if not use_batch_norm else None,
          weights_initializer=hidden_weights_initializer,
          biases_initializer=hidden_biases_initializer,
          scope='hid%d' % idx,
        )

        if use_batch_norm and idx != action_merge_layer:
          hidden_layer = hidden_activation_fn(batch_norm(hidden_layer, is_train))

      def make_output(layer, num_outputs, activation_fn=None, scope='out'):
        return fully_connected(
          layer,
          num_outputs=num_outputs,
          activation_fn=output_activation_fn if activation_fn == None else activation_fn,
          weights_initializer=output_weights_initializer,
          biases_initializer=output_biases_initializer,
          scope=scope,
        )

      with tf.variable_scope('advantage'):
        l = make_output(hidden_layer, (action_size * (action_size + 1))/2, scope='l')
        mu = make_output(hidden_layer, action_size, scope='mu')

        columns = []
        for idx in xrange(action_size):
          column = tf.pad(tf.slice(l, (0, 0), (-1, action_size - idx)), ((0, 0), (idx, 0)))
          columns.append(column)

        L = tf.pack(columns, axis=1)
        P = tf.batch_matmul(L, tf.transpose(L, (0, 2, 1)))

        tmp = tf.expand_dims(u - mu, 2)
        A = -tf.batch_matmul(tf.transpose(tmp, [0, 2, 1]), tf.batch_matmul(P, tmp))/2
        A = tf.reshape(A, [-1, 1])

      with tf.variable_scope('value'):
        V = make_output(hidden_layer, 1, scope='V')

      Q = A + V

      with tf.variable_scope('optimizer'):
        target_Q = tf.placeholder(tf.float32, [None], name='target_Q')
        loss = tf.reduce_mean(tf.square(target_Q - Q), name='loss')

      self.x = x
      self.u = u
      self.loss = loss
      self.target_Q = target_Q
      self.is_train = is_train

      self.V, self.L, self.P, self.mu, self.A, self.Q = V, L, P, mu, A, Q
      self.variables = get_variables(name)

  def predict(self, state):
    return self.sess.run(self.mu, {self.x: state, self.is_train: False})

  def make_copy_from(self, network):
    self.assign_op = {}

    for from_, to_ in zip(network.variables, self.variables):
      self.assign_op[to_.name] = to_.assign(from_)

  def update_from(self, network):
    for variable in self.variables:
      self.sess.run(self.assign_op[variable.name])
