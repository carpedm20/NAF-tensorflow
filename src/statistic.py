import os
import numpy as np
import tensorflow as tf
from logging import getLogger

logger = getLogger(__name__)

class Statistic(object):
  def __init__(self, sess, env_name, model_dir, variables, max_update_per_step, max_to_keep=20):
    self.sess = sess
    self.env_name = env_name
    self.max_update_per_step = max_update_per_step

    self.reset()
    self.max_avg_r = None

    with tf.variable_scope('t'):
      self.t_op = tf.Variable(0, trainable=False, name='t')
      self.t_add_op = self.t_op.assign_add(1)

    self.model_dir = model_dir
    self.saver = tf.train.Saver(variables + [self.t_op], max_to_keep=max_to_keep)
    self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['total r', 'avg r', 'avg q', 'avg v', 'avg a', 'avg l']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar('%s/%s' % (self.env_name, tag), self.summary_placeholders[tag])

  def reset(self):
    self.total_q = 0.
    self.total_v = 0.
    self.total_a = 0.
    self.total_l = 0.

    self.ep_step = 0
    self.ep_rewards = []

  def on_step(self, action, reward, terminal, q, v, a, l):
    self.t = self.t_add_op.eval(session=self.sess)

    self.total_q += q
    self.total_v += v
    self.total_a += a
    self.total_l += l

    self.ep_step += 1
    self.ep_rewards.append(reward)

    if terminal:
      avg_q = self.total_q / self.ep_step / self.max_update_per_step
      avg_v = self.total_v / self.ep_step / self.max_update_per_step
      avg_a = self.total_a / self.ep_step / self.max_update_per_step
      avg_l = self.total_l / self.ep_step / self.max_update_per_step

      avg_r = np.mean(self.ep_rewards)
      total_r = np.sum(self.ep_rewards)

      logger.info('t: %d, R: %.3f, r: %.3f, q: %.3f, v: %.3f, a: %.3f, l: %.3f' \
          % (self.t, total_r, avg_r, avg_q, avg_q, avg_a, avg_l))

      if self.max_avg_r == None:
        self.max_avg_r = avg_r

      if self.max_avg_r * 0.9 <= avg_r:
        self.save_model(self.t)
        self.max_avg_r = max(self.max_avg_r, avg_r)

      self.inject_summary({
        'total r': total_r, 'avg r': avg_r,
        'avg q': avg_q, 'avg v': avg_v, 'avg a': avg_a, 'avg l': avg_l,
      }, self.t)

      self.reset()

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)

  def save_model(self, t):
    logger.info("Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.saver.save(self.sess, self.model_dir, global_step=t)

  def load_model(self):
    logger.info("Loading checkpoints...")
    tf.initialize_all_variables().run()

    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.model_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      logger.info("Load SUCCESS: %s" % fname)
    else:
      logger.info("Load FAILED: %s" % self.model_dir)

    self.t = self.t_add_op.eval(session=self.sess)
