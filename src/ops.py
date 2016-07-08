from tensorflow.contrib.layers import fully_connected, initializers

from tensorflow.python import control_flow_ops

def batch_norm(x, phase_train, 
    n_out=[0], epsilon=1e-4, alpha=0.1, scope='bn'):
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma')

    mean, variance = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(
      phase_train,
      mean_var_with_update,
      lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
  return normed
