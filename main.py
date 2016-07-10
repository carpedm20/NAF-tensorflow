import random
import tensorflow as tf

from src.naf import NAF

flags = tf.app.flags

flags.DEFINE_integer('memory_size', 100000, '')
flags.DEFINE_float('noise', 0.1, 'The value of noise')
flags.DEFINE_string('env_name', 'BipedalWalker-v2', 'The name of environment')

# training
flags.DEFINE_integer('max_update', 10, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_step', 10000, '')
flags.DEFINE_integer('learn_start', 10, '')
flags.DEFINE_integer('num_train', 100000, '')
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
  with tf.Session() as sess:
    agent = NAF(sess, config.env_name, config.memory_size, config.batch_size)

    if config.is_train:
      agent.train(config.num_train,
                  config.learning_rate,
                  config.learn_start,
                  config.max_step,
                  config.max_update,
                  config.monitor,
                  config.display)
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
