import tensorflow as tf
from tensorflow.contrib.framework import get_variables
from tensorflow.contrib.layers import fully_connected, initializers, l1_regularizer, l2_regularizer, batch_norm


from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Code modification of tensorflow/contrib/layers/python/layers/layers.py
  """
  with variable_scope.variable_op_scope([inputs],
                                        scope, 'BatchNorm', reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(inputs_rank - 1))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_mean_collections = utils.get_variable_collections(
        variables_collections, 'moving_mean')
    moving_mean = variables.model_variable(
        'moving_mean',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.zeros_initializer,
        trainable=False,
        collections=moving_mean_collections)
    moving_variance_collections = utils.get_variable_collections(
        variables_collections, 'moving_variance')
    moving_variance = variables.model_variable(
        'moving_variance',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.ones_initializer,
        trainable=False,
        collections=moving_variance_collections)

    # Calculate the moments based on the individual batch.
    mean, variance = nn.moments(inputs, axis, shift=moving_mean)
    # Update the moving_mean and moving_variance moments.
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
    if updates_collections is None:
      # Make sure the updates are computed here.
      with ops.control_dependencies([update_moving_mean,
                                      update_moving_variance]):
        outputs = nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
    else:
      # Collect the updates to be computed later.
      ops.add_to_collections(updates_collections, update_moving_mean)
      ops.add_to_collections(updates_collections, update_moving_variance)
      outputs = nn.batch_normalization(
          inputs, mean, variance, beta, gamma, epsilon)

    test_outputs = nn.batch_normalization(
        inputs, moving_mean, moving_variance, beta, gamma, epsilon)

    outputs = tf.cond(is_training, lambda: outputs, lambda: test_outputs)
    outputs.set_shape(inputs_shape)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

"""
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
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    beta = tf.Variable(tf.constant(0.0, shape=shape), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=shape), name='gamma')

    mean, variance = tf.nn.moments(layer, axes)
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
"""
