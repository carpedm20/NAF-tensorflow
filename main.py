import random
import logging
import tensorflow as tf

from src.naf import NAF
from utils import get_model_dir

flags = tf.app.flags

# memory, environment, network
flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of environment')
flags.DEFINE_integer('memory_size', 10**6, 'size of memory')
flags.DEFINE_boolean('use_batch_norm', False, 'use batch normalization or not')
flags.DEFINE_float('l1_reg_scale', None, 'scale of l1 regularization')
flags.DEFINE_float('l2_reg_scale', 0.01, 'scale of l2 regularization')
flags.DEFINE_string('hidden_dims', '[100, 100]', 'dimension of hidden layers')
flags.DEFINE_string('hidden_activation_fn', 'tanh', 'type of activation function of hidden layer [tanh, relu]')

# training
flags.DEFINE_string('noise', 'linear_decay', 'type of noise')
flags.DEFINE_float('noise_scale', 0.01, 'scale of noise')
flags.DEFINE_float('discount', 0.99, 'discount factor of Q-learning')
flags.DEFINE_float('learning_rate', 1e-3, 'value of learning rate')
flags.DEFINE_float('decay', 0.99, 'decay for moving average')
flags.DEFINE_float('epsilon', 1e-4, 'epsilon for batch normalization')
flags.DEFINE_float('tau', 0.001, 'tau of soft target update')

# test
flags.DEFINE_integer('max_update', 10, 'maximum # of q-learning update for each step')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('test_step', 100, '# of episode interval to run test')
flags.DEFINE_integer('max_step', 100000, 'maximum step for each episode')
flags.DEFINE_integer('max_episode', 200, 'maximum # of episode to train')

# misc.
flags.DEFINE_boolean('is_train', True, 'Training or Test')
flags.DEFINE_integer('random_seed', 123, 'The value of random seed')
flags.DEFINE_boolean('monitor', False, 'Whether to monitor the training or not')
flags.DEFINE_boolean('display', False, 'Whether to display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS

# logger
logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)

def main(_):
  conf.hidden_dims = eval(conf.hidden_dims)

  model_dir = get_model_dir(conf,
      ['test_step', 'max_step', 'max_episode',
       'is_train', 'random_seed', 'monitor', 'display', 'log_level'])

  if conf.hidden_activation_fn == 'tanh':
    conf.hidden_activation_fn = tf.nn.tanh
  elif conf.hidden_activation_fn == 'relu':
    conf.hidden_activation_fn = tf.nn.relu
  else:
    raise Exception('Unknown hidden_activation_fn: %s' % conf.hidden_activation_fn)
       
  with tf.Session() as sess:
    agent = NAF(sess, model_dir, conf.env_name,
                conf.use_batch_norm, conf.l1_reg_scale, conf.l2_reg_scale,
                conf.hidden_dims, conf.hidden_activation_fn,
                conf.noise, conf.noise_scale,
                conf.tau, conf.decay, conf.epsilon, conf.discount,
                conf.memory_size, conf.batch_size,
                conf.learning_rate,
                conf.max_step, conf.max_update, conf.max_episode, conf.test_step)

    if conf.is_train:
      agent.train(conf.monitor, conf.display)
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
