import gym
import random
import logging
import tensorflow as tf

from src.naf import NAF
from src.network import Network
from src.statistic import Statistic
from src.exploration import OUExploration, BrownianExploration
from utils import get_model_dir

flags = tf.app.flags

# memory, environment, network
flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of environment')
flags.DEFINE_boolean('use_batch_norm', False, 'use batch normalization or not')
flags.DEFINE_boolean('use_seperate_networks', True, 'use seperate networks for mu, V and A')
flags.DEFINE_boolean('clip_action', False, 'whether to clip an action with given bound')
flags.DEFINE_float('l1_reg_scale', None, 'scale of l1 regularization')
flags.DEFINE_float('l2_reg_scale', 0.001, 'scale of l2 regularization')
flags.DEFINE_string('hidden_dims', '[100, 100]', 'dimension of hidden layers')
flags.DEFINE_string('hidden_activation_fn', 'tanh', 'type of activation function of hidden layer [tanh, relu]')

# training
flags.DEFINE_string('noise', 'ou', 'type of noise exploration [brownian, ou]')
flags.DEFINE_float('noise_scale', 0.3, 'scale of noise')
flags.DEFINE_float('discount', 0.99, 'discount factor of Q-learning')
flags.DEFINE_float('learning_rate', 1e-3, 'value of learning rate')
flags.DEFINE_float('decay', 0.99, 'decay for moving average')
flags.DEFINE_float('epsilon', 1e-4, 'epsilon for batch normalization')
flags.DEFINE_float('tau', 0.001, 'tau of soft target update')

# test
flags.DEFINE_integer('max_update', 5, 'maximum # of q-learning update for each step')
flags.DEFINE_integer('batch_size', 150, 'batch size')
flags.DEFINE_integer('max_step', 200, 'maximum step for each episode')
flags.DEFINE_integer('max_episode', 1000, 'maximum # of episode to train')

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
      ['max_step', 'max_episode',
       'is_train', 'random_seed', 'monitor', 'display', 'log_level'])

  if conf.hidden_activation_fn == 'tanh':
    conf.hidden_activation_fn = tf.nn.tanh
  elif conf.hidden_activation_fn == 'relu':
    conf.hidden_activation_fn = tf.nn.relu
  else:
    raise Exception('Unknown hidden_activation_fn: %s' % conf.hidden_activation_fn)
       
  with tf.Session() as sess:
    # environment
    env = gym.make(conf.env_name)

    assert isinstance(env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    # exploration strategy
    if conf.noise == 'ou':
      strategy = OUExploration(env, sigma=conf.noise_scale, clip_action=conf.clip_action)
    elif conf.noise == 'brownian':
      strategy = BrownianExploration(env, conf.noise_scale, clip_action=conf.clip_action)
    else:
      raise ValueError('Unkown exploration strategy: %s' % conf.noise)

    # networks
    shared_args = {
      'session': sess,
      'input_shape': env.observation_space.shape,
      'action_size': env.action_space.shape[0],
      'use_batch_norm': conf.use_batch_norm,
      'use_seperate_networks': conf.use_seperate_networks,
      'l1_reg_scale': conf.l1_reg_scale, 'l2_reg_scale': conf.l2_reg_scale,
      'hidden_dims': conf.hidden_dims, 'hidden_activation_fn': conf.hidden_activation_fn,
      'decay': conf.decay, 'epsilon': conf.epsilon,
    }

    logger.info("Creating prediction network...")
    pred_network = Network(
      name='pred_network', **shared_args
    )

    logger.info("Creating target network...")
    target_network = Network(
      name='target_network', **shared_args
    )
    target_network.make_soft_update_from(pred_network, conf.tau)

    # statistic
    stat = Statistic(sess, conf.env_name, conf.max_update, model_dir, pred_network.variables)

    agent = NAF(env, strategy, pred_network, target_network, stat,
                conf.discount, conf.batch_size, conf.learning_rate,
                conf.max_step, conf.max_update, conf.max_episode)

    agent.run(conf.monitor, conf.display, conf.is_train)

if __name__ == '__main__':
  tf.app.run()
