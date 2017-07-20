# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:31:24 2017

@author: Zhenqin Wu
"""

import torch
import time
import numpy as np
import itertools
from deepchem.models import Model

class TorchModel(Model):

  def __init__(self,
               n_tasks=1,
               learning_rate=.001,
               momentum=.9,
               optimizer="adam",
               batch_size=16,
               pad_batches=False,
               seed=None):
    """Abstract class for Torch models

    Parameters
    ----------
    n_tasks: int, optional
      Number of tasks
    learning_rate: float, optional
      Learning rate for model.
    momentum: float, optional
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str, optional
      Type of optimizer applied.
    batch_size: int, optional
      Size of minibatches for training.GraphConv
    pad_batches: bool, optional
      Perform logging.
    seed: int, optional
      If not none, is used as random seed for tensorflow.
    """
    # Save hyperparameters
    self.n_tasks = n_tasks
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.pad_batches = pad_batches
    self.seed = seed
    self.trainable_layers = []
    self.build()
    self.optimizer = self.get_training_op()
    self.regularizaed_variables = []

  def get_training_op(self):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    An optimizer
    """
    trainables = [layer.parameters() for layer in self.trainable_layers]
    self.trainables = itertools.chain(*trainables)
    if self.optimizer == "adam":
      train_op = torch.optim.Adam(self.trainables, lr=self.learning_rate)
    elif self.optimizer == 'adagrad':
      train_op = torch.optim.Adagrad(self.trainables, lr=self.learning_rate)
    elif self.optimizer == 'rmsprop':
      train_op = torch.optim.RMSprop(
          self.trainables, lr=self.learning_rate, momentum=self.momentum)
    elif self.optimizer == 'sgd':
      train_op = torch.optim.SGD(self.trainables, lr=self.learning_rate)
    else:
      raise NotImplementedError('Unsupported optimizer %s' % self.optimizer)
    return train_op

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=5,
          checkpoint_interval=100):
    return self.fit_generator(
        self.default_generator(dataset, epochs=nb_epoch),
        max_checkpoints_to_keep, log_every_N_batches=log_every_N_batches,
        checkpoint_interval=checkpoint_interval)

  def fit_generator(self,
          generator,
          max_checkpoints_to_keep=5,
          log_every_N_batches=5,
          checkpoint_interval=10):
    """Fit the model.

    Parameters
    ----------
    generator: generator object
      generate batch of data
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    avg_loss, n_batches = 0., 0
    for inputs, labels, weights in generator:
      if n_batches % checkpoint_interval == 0 and n_batches > 0:
        loss_out = float(avg_loss.data.cpu().numpy()) / n_batches
        print("On batch %d, loss %f" % (n_batches, loss_out))
      # Run training op.
      self.optimizer.zero_grad()
      outputs = self.forward(inputs, training=True)
      loss = self.add_training_cost(outputs, labels, weights)
      loss.backward()
      self.optimizer.step()
      avg_loss += loss
      n_batches += 1
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

  def add_training_cost(self, outputs, labels, weights):
    weighted_costs = []  # weighted costs for each example
    for task in range(self.n_tasks):
      weighted_cost = self.cost(outputs[task], labels[:, task],
                                weights[:, task])
      weighted_costs.append(weighted_cost)
    loss = torch.cat(weighted_costs).sum()
    # weight decay
    if self.penalty > 0.0:
      for variable in self.regularizaed_variables:
        loss += self.penalty * 0.5 * variable.mul(variable).sum()
    return loss
    
  def predict(self, dataset):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator)

  def predict_proba(self, dataset):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator)

  def predict_on_generator(self, generator):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    n_tasks = self.n_tasks
    for inputs, labels, weights in generator:
      y_pred_batch = self.predict_on_batch(inputs)
      assert y_pred_batch.shape[1] == n_tasks
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds, axis=0)
    return y_pred

  def predict_proba_on_generator(self, generator):
    y_preds = []
    n_tasks = self.n_tasks
    for inputs, labels, weights in generator:
      y_pred_batch = self.predict_proba_on_batch(inputs)
      assert y_pred_batch.shape[1] == n_tasks
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds, axis=0)
    return y_pred

  def build(self):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def forward(self, X, training=False):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def cost(self, logit, label, weight):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def predict_on_batch(self, X_batch):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def predict_proba_on_batch(self, X_batch):
    raise NotImplementedError('Must be overridden by concrete subclass')
