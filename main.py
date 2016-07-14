import gym
import logging
import numpy as np
import tensorflow as tf

from src.naf import NAF
from src.network import Network
from src.statistic import Statistic
from src.exploration import OUExploration, BrownianExploration, LinearDecayExploration
from utils import get_model_dir, preprocess_conf

flags = tf.app.flags

# environment
flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of environment')

# network
flags.DEFINE_string('hidden_dims', '[200, 200]', 'dimension of hidden layers')
flags.DEFINE_boolean('use_batch_norm', False, 'use batch normalization or not')
flags.DEFINE_boolean('clip_action', False, 'whether to clip an action with given bound')
flags.DEFINE_boolean('use_seperate_networks', False, 'use seperate networks for mu, V and A')
flags.DEFINE_string('hidden_w', 'uniform_big', 'weight initialization of hidden layers [uniform_small, uniform_big, he]')
flags.DEFINE_string('hidden_fn', 'tanh', 'activation function of hidden layer [none, tanh, relu]')
flags.DEFINE_string('action_w', 'uniform_big', 'weight initilization of action layer [uniform_small, uniform_big, he]')
flags.DEFINE_string('action_fn', 'tanh', 'activation function of action layer [none, tanh, relu]')
flags.DEFINE_string('w_reg', 'none', 'weight regularization [none, l1, l2]')
flags.DEFINE_float('w_reg_scale', 0.001, 'scale of regularization')

# exploration
flags.DEFINE_float('noise_scale', 0.3, 'scale of noise')
flags.DEFINE_string('noise', 'ou', 'type of noise exploration [ou, linear_decay, brownian]')

# training
flags.DEFINE_float('tau', 0.001, 'tau of soft target update')
flags.DEFINE_float('discount', 0.99, 'discount factor of Q-learning')
flags.DEFINE_float('learning_rate', 1e-4, 'value of learning rate')
flags.DEFINE_integer('batch_size', 100, 'The size of batch for minibatch training')
flags.DEFINE_integer('max_steps', 150, 'maximum # of steps for each episode')
flags.DEFINE_integer('update_repeat', 5, 'maximum # of q-learning updates for each step')
flags.DEFINE_integer('max_episodes', 10000, 'maximum # of episodes to train')

# Debug
flags.DEFINE_boolean('is_train', True, 'training or testing')
flags.DEFINE_integer('random_seed', 123, 'random seed')
flags.DEFINE_boolean('monitor', False, 'monitor the training or not')
flags.DEFINE_boolean('display', False, 'display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

# set random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

def main(_):
  model_dir = get_model_dir(conf, 
      ['is_train', 'random_seed', 'monitor', 'display', 'log_level'])
  preprocess_conf(conf)

  with tf.Session() as sess:
    # environment
    env = gym.make(conf.env_name)
    env._seed(conf.random_seed)

    assert isinstance(env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    # exploration strategy
    if conf.noise == 'ou':
      strategy = OUExploration(env, sigma=conf.noise_scale)
    elif conf.noise == 'brownian':
      strategy = BrownianExploration(env, conf.noise_scale)
    elif conf.noise == 'linear_decay':
      strategy = LinearDecayExploration(env)
    else:
      raise ValueError('Unkown exploration strategy: %s' % conf.noise)

    # networks
    shared_args = {
      'sess': sess,
      'input_shape': env.observation_space.shape,
      'action_size': env.action_space.shape[0],
      'hidden_dims': conf.hidden_dims, 
      'use_batch_norm': conf.use_batch_norm,
      'use_seperate_networks': conf.use_seperate_networks,
      'hidden_w': conf.hidden_w, 'action_w': conf.action_w,
      'hidden_fn': conf.hidden_fn, 'action_fn': conf.action_fn,
      'w_reg': conf.w_reg,
    }

    logger.info("Creating prediction network...")
    pred_network = Network(
      scope='pred_network', **shared_args
    )

    logger.info("Creating target network...")
    target_network = Network(
      scope='target_network', **shared_args
    )
    target_network.make_soft_update_from(pred_network, conf.tau)

    # statistic
    stat = Statistic(sess, conf.env_name, model_dir, pred_network.variables, conf.update_repeat)

    agent = NAF(sess, env, strategy, pred_network, target_network, stat,
                conf.discount, conf.batch_size, conf.learning_rate,
                conf.max_steps, conf.update_repeat, conf.max_episodes)

    agent.run(conf.monitor, conf.display, conf.is_train)

if __name__ == '__main__':
  tf.app.run()
