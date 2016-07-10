import random
import tensorflow as tf

from src.naf import NAF
from utils import get_model_dir

flags = tf.app.flags

flags.DEFINE_integer('memory_size', 100000, '')
flags.DEFINE_float('noise', 0.1, 'The value of noise')
flags.DEFINE_float('discount', 0.99, 'The discount factor of Q-learning')
flags.DEFINE_string('env_name', 'BipedalWalker-v2', 'The name of environment')

# training
flags.DEFINE_integer('max_update', 10, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('learn_start', 100, '')
flags.DEFINE_integer('max_step', 200, '')
flags.DEFINE_integer('max_episode', 100000, '')
flags.DEFINE_integer('target_q_update_step', 1000, '')
flags.DEFINE_float('learning_rate', 1e-4, 'The value of learning rate')

# misc.
flags.DEFINE_boolean('is_train', True, 'Training or Test')
flags.DEFINE_integer('random_seed', 123, 'The value of random seed')
flags.DEFINE_boolean('monitor', False, 'Whether to monitor the training or not')
flags.DEFINE_boolean('display', False, 'Whether to display the game screen or not')

config = flags.FLAGS

# set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

def main(_):
  model_dir = get_model_dir(config, ['is_train', 'random_seed', 'monitor', 'display'])

  with tf.Session() as sess:
    agent = NAF(sess,
                model_dir,
                config.env_name,
                config.discount,
                config.memory_size,
                config.batch_size,
                config.learning_rate,
                config.learn_start,
                config.max_step,
                config.max_update,
                config.max_episode)

    if config.is_train:
      agent.train(config.monitor, config.display)
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
