from logging import getLogger
logger = getLogger(__name__)

import tensorflow as tf
from tensorflow.contrib.framework import get_variables

from .ops import *

class Network:
  def __init__(self, sess, input_shape, action_size, hidden_dims,
               use_batch_norm, use_seperate_networks,
               hidden_w, action_w, hidden_fn, action_fn, w_reg,
               scope='NAF'):
    self.sess = sess
    with tf.variable_scope(scope):
      x = tf.placeholder(tf.float32, (None,) + tuple(input_shape), name='observations')
      u = tf.placeholder(tf.float32, (None, action_size), name='actions')
      is_train = tf.placeholder(tf.bool, name='is_train')

      hid_outs = {}
      with tf.name_scope('hidden'):
        if use_seperate_networks:
          logger.info("Creating seperate networks for v, l, and mu")

          for scope in ['v', 'l', 'mu']:
            with tf.variable_scope(scope):
              if use_batch_norm:
                h = batch_norm(x, is_training=is_train)
              else:
                h = x

              for idx, hidden_dim in enumerate(hidden_dims):
                h = fc(h, hidden_dim, is_train, hidden_w, weight_reg=w_reg,
                       activation_fn=hidden_fn, use_batch_norm=use_batch_norm, scope='hid%d' % idx)
              hid_outs[scope] = h
        else:
          logger.info("Creating shared networks for v, l, and mu")

          if use_batch_norm:
            h = batch_norm(x, is_training=is_train)
          else:
            h = x

          for idx, hidden_dim in enumerate(hidden_dims):
            h = fc(h, hidden_dim, is_train, hidden_w, weight_reg=w_reg,
                   activation_fn=hidden_fn, use_batch_norm=use_batch_norm, scope='hid%d' % idx)
          hid_outs['v'], hid_outs['l'], hid_outs['mu'] = h, h, h

      with tf.name_scope('value'):
        V = fc(hid_outs['v'], 1, is_train,
               hidden_w, use_batch_norm=use_batch_norm, scope='V')

      with tf.name_scope('advantage'):
        l = fc(hid_outs['l'], (action_size * (action_size + 1))/2, is_train, hidden_w,
               use_batch_norm=use_batch_norm, scope='l')
        mu = fc(hid_outs['mu'], action_size, is_train, action_w,
                activation_fn=action_fn, use_batch_norm=use_batch_norm, scope='mu')

        pivot = 0
        rows = []
        for idx in xrange(action_size):
          count = action_size - idx

          diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
          non_diag_elems = tf.slice(l, (0, pivot+1), (-1, count-1))
          row = tf.pad(tf.concat(1, (diag_elem, non_diag_elems)), ((0, 0), (idx, 0)))
          rows.append(row)

          pivot += count

        L = tf.transpose(tf.pack(rows, axis=1), (0, 2, 1))
        P = tf.batch_matmul(L, tf.transpose(L, (0, 2, 1)))

        tmp = tf.expand_dims(u - mu, -1)
        A = -tf.batch_matmul(tf.transpose(tmp, [0, 2, 1]), tf.batch_matmul(P, tmp))/2
        A = tf.reshape(A, [-1, 1])

      with tf.name_scope('Q'):
        Q = A + V

      with tf.name_scope('optimization'):
        self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(Q)), name='loss')

    self.is_train = is_train
    self.variables = get_variables(scope)

    self.x, self.u, self.mu, self.V, self.Q, self.P, self.A = x, u, mu, V, Q, P, A

  def predict_v(self, x, u):
    return self.sess.run(self.V, {
      self.x: x, self.u: u, self.is_train: False,
    })

  def predict(self, state):
    return self.sess.run(self.mu, {
      self.x: state, self.is_train: False
    })

  def update(self, optim, target_v, x_t, u_t):
    _, q, v, a, l = self.sess.run([
        optim, self.Q, self.V, self.A, self.loss
      ], {
        self.target_y: target_v,
        self.x: x_t,
        self.u: u_t,
        self.is_train: True,
      })
    return q, v, a, l

  def make_soft_update_from(self, network, tau):
    logger.info("Creating ops for soft target update...")
    assert len(network.variables) == len(self.variables), \
      "target and prediction network should have same # of variables"

    self.assign_op = {}
    for from_, to_ in zip(network.variables, self.variables):
      if 'BatchNorm' in to_.name:
        self.assign_op[to_.name] = to_.assign(from_)
      else:
        self.assign_op[to_.name] = to_.assign(tau * from_ + (1-tau) * to_)

  def hard_copy_from(self, network):
    logger.info("Creating ops for hard target update...")
    assert len(network.variables) == len(self.variables), \
      "target and prediction network should have same # of variables"

    for from_, to_ in zip(network.variables, self.variables):
      self.sess.run(to_.assign(from_))

  def soft_update_from(self, network):
    for variable in self.variables:
      self.sess.run(self.assign_op[variable.name])
    return True
