import os
import pprint
import tensorflow as tf

from src.network import *

pp = pprint.PrettyPrinter().pprint

def get_model_dir(config, exceptions=None):

  attrs = config.__flags
  pp(attrs)

  keys = attrs.keys()
  keys.sort()
  keys.remove('env_name')
  keys = ['env_name'] + keys

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags

  for option, value in options.items():
    option = option.lower()
    value = value.value

    if option == 'hidden_dims':
      conf.hidden_dims = eval(conf.hidden_dims)
    elif option == 'w_reg':
      if value == 'l1':
        w_reg = l1_regularizer(conf.w_reg_scale)
      elif value == 'l2':
        w_reg = l2_regularizer(conf.w_reg_scale)
      elif value == 'none':
        w_reg = None
      else:
        raise ValueError('Wrong weight regularizer %s: %s' % (option, value))
      conf.w_reg = w_reg
    elif option.endswith('_w'):
      if value == 'uniform_small':
        weights_initializer = random_uniform_small
      elif value == 'uniform_big':
        weights_initializer = random_uniform_big
      elif value == 'he':
        weights_initializer = he_uniform
      else:
        raise ValueError('Wrong %s: %s' % (option, value))
      setattr(conf, option, weights_initializer)
    elif option.endswith('_fn'):
      if value == 'tanh':
        activation_fn = tf.nn.tanh
      elif value == 'relu':
        activation_fn = tf.nn.relu
      elif value == 'none':
        activation_fn = None
      else:
        raise ValueError('Wrong %s: %s' % (option, value))
      setattr(conf, option, activation_fn)
