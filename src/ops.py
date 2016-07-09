import tensorflow as tf
from tensorflow.python import control_flow_ops
from tensorflow.contrib.layers import fully_connected, initializers

def batch_norm(layer,
               is_train, 
               decay=0.9, epsilon=1e-4,
               data_format='NCHW', scope='bn'):
  input_shape = layer.get_shape().as_list()

  if data_format == 'NCHW':
    axes = (0,) + tuple(range(2, len(input_shape)))
  elif data_format == 'NHWC':
    axes = (0,) + tuple(range(1, len(input_shape) - 1))
  else:
    raise ValueError('Unknown data_format: %s' % data_format)

  shape = [size for axis, size in enumerate(input_shape) if axis not in axes]

  with tf.variable_scope(scope):
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    beta = tf.Variable(tf.constant(0.0, shape=shape), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=shape), name='gamma')

    mean, variance = tf.nn.moments(layer, axes, name='moments')
    maintain_averages_op = ema.apply([mean, variance])

    with tf.control_dependencies([maintain_averages_op]):
      mean_with_dep, variance_with_dep = tf.identity(mean), tf.identity(variance)

    shadow_mean = ema.average(mean)
    shadow_variance = ema.average(variance)

    m, v = tf.cond(is_train,
                   lambda: (mean_with_dep, variance_with_dep),
                   lambda: (shadow_mean, shadow_variance))

    return tf.nn.batch_normalization(
      layer, m, v, beta, gamma, epsilon)
