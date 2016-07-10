import os
import tensorflow as tf

class Agent(object):
  def save_model(self, t):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.saver.save(self.sess, self.model_dir, global_step=t)

  def load_model(self):
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.model_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.model_dir)
      return False
