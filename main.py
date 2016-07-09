import gym
import random
import tensorflow as tf

from src.naf import NAF

flags = tf.app.flags
flags.DEFINE_string('env', 'LunarLander-v2', 'The name of environment')
flags.DEFINE_integer('random_seed', 123, 'The value of random seed')
flags.DEFINE_integer('num_train', 100000, '')
flags.DEFINE_integer('learn_start', 100, '')
flags.DEFINE_float('learning_rate', 1e-4, 'The value of learning rate')
flags.DEFINE_float('noise', 0.1, 'The value of noise')

flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')

config = flags.FLAGS

# set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

def main(_):
  env = gym.make(config.env)
  assert isinstance(env.observation_space, gym.spaces.Box), "observation space must be continuous"

  with tf.Session() as sess:
    agent = NAF(env, sess)

    if config.is_train:
      agent.train(config.num_train, config.learning_rate, config.learn_start)
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
