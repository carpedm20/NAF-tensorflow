import random
import logging
import tensorflow as tf

from src.naf import NAF
from utils import get_model_dir

flags = tf.app.flags

# memory, environment, network
flags.DEFINE_integer('memory_size', 10**6, '')
flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of environment')
flags.DEFINE_string('hidden_dims', '[100, 100]', 'dimension of hidden layers')

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
flags.DEFINE_integer('max_step', 200, 'maximum step for each episode')
flags.DEFINE_integer('max_episode', 200, 'maximum # of episode to train')

# misc.
flags.DEFINE_boolean('is_train', True, 'Training or Test')
flags.DEFINE_integer('random_seed', 123, 'The value of random seed')
flags.DEFINE_boolean('monitor', False, 'Whether to monitor the training or not')
flags.DEFINE_boolean('display', False, 'Whether to display the game screen or not')
flags.DEFINE_string('log_level', 'DEBUG', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

config = flags.FLAGS

# logger
logger = logging.getLogger()
logger.propagate = False
logger.setLevel(config.log_level)

# set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

def main(_):
  config.hidden_dims = eval(config.hidden_dims)

  model_dir = get_model_dir(config,
      ['test_step', 'max_step', 'max_episode',
       'is_train', 'random_seed', 'monitor', 'display', 'log_level'])
       
  with tf.Session() as sess:
    agent = NAF(sess, model_dir, config.env_name, config.hidden_dims,
                config.noise, config.noise_scale,
                config.tau, config.decay, config.epsilon, config.discount,
                config.memory_size, config.batch_size,
                config.learning_rate,
                config.max_step, config.max_update, config.max_episode, config.test_step)

    if config.is_train:
      agent.train(config.monitor, config.display)
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
