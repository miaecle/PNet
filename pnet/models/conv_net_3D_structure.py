#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:28:44 2017
@author: zqwu
"""

import numpy as np
import tensorflow as tf
import pickle
import time
import threading

import deepchem as dc
from rdkit import Chem
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, TensorWrapper, GraphConv, GraphPool, GraphGather, Add, \
    Reshape, Squeeze
from deepchem.models.tensorgraph.optimizers import Adam
from pnet.models.layers import BatchNorm, AminoAcidEmbedding, AminoAcidPad, \
    Conv1DLayer, Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, \
    WeightedL2Loss, AddThreshold, SigmoidLoss, Sigmoid, TriangleInequality, \
    CoordinatesToDistanceMap
from pnet.utils.amino_acids import AminoAcid_SMILES
from pnet.models.conv_net_contact_map import ConvNetContactMapBase

class ConvNet3DStructureBase(ConvNetContactMapBase):
  """Base Class for Convolutional network contact map prediction"""
  def __init__(self,
               n_1D_feat=50,
               n_2D_feat=1,
               **kwargs):
    """
    Parameters:
    -----------
    n_1D_feat: int
      number of features for 1d inputs
    n_2D_feat: int
      number of features for 2d inputs
    uppertri: bool, optional
      build symmetry matrix(False) or upper-triangular matrix(True)
    """
    self.n_1D_feat = n_1D_feat
    self.n_2D_feat = n_2D_feat
    super(ConvNet3DStructureBase, self).__init__(**kwargs)
    
    
  def build_graph(self):
    """ Build graph structure """
    with self._get_tf("Graph").as_default():
      self.input_1D_features = Feature(shape=(self.batch_size, None, self.n_1D_feat), name='1D_features')
      self.input_2D_features = Feature(shape=(self.batch_size, None, None, self.n_2D_feat), name='2D_features')
      self.training_placeholder = Feature(shape=(), dtype=tf.bool, name='training_placeholder')
      # Placeholder for valid index
      self.res_flag_1D = Feature(shape=(self.batch_size, None), dtype=tf.int32, name='flag_1D')
      self.res_flag_2D = Feature(shape=(self.batch_size, None, None), dtype=tf.int32, name='flag_2D')
      self.n_residues = Feature(shape=(self.batch_size,), dtype=tf.int32, name='n_res')
      
      n_input = self.n_2D_feat
      in_layer = self.input_2D_features

      # User-defined structures
      n_input, self.conv2d_out_layer = self.Conv2DModule(n_input, in_layer)
      n_input, self.condense_layer = self.CondenseModule(n_input, self.conv2d_out_layer)
      
      
      n_input, self.conv1d_out_layer = self.Conv1DModule([n_input, self.n_1D_feat],
                                                         [self.condense_layer_layer, self.input_1D_features])

      n_out, self.cost = self.LossModule(n_input, self.conv1d_out_layer)
      self.set_loss(self.cost)      
      return 

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True, 
                        **kwargs):
    # TODO: change this
    """ Transform each batch into corresponding feed_dict """
    for epoch in range(epochs):
      for (X_b, twoD_X_b, y_b, w_b, oneD_y_b, oneD_w_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):
        
        n_residues = [seq_feat.shape[0] for seq_feat in X_b]
        max_n_res = max(n_residues)
        
        feed_dict = dict()
        if not self.training_placeholder.out_tensor is None:
          feed_dict[self.training_placeholder] = not predict
        
        if not y_b is None and not predict and not self.contact_labels.out_tensor is None:
          labels = []
          for ids, label in enumerate(y_b):
            if self.uppertri:
              label_eff = [label[i, i:] for i in range(label.shape[0])]
              labels.append(np.concatenate(label_eff, 0))
            else:
              labels.append(label.flatten())
          if self.mode == "classification":
            feed_dict[self.contact_labels] = to_one_hot(np.concatenate(labels, axis=0))
          elif self.mode == "regression":
            feed_dict[self.contact_labels] = np.reshape(np.concatenate(labels, axis=0), (-1, 1))
        
        if not w_b is None and not predict and not self.contact_weights.out_tensor is None:
          weights = []
          for ids, weight in enumerate(w_b):
            if self.uppertri:
              weight_eff = [weight[i, i:] for i in range(weight.shape[0])]
              weights.append(np.concatenate(weight_eff, 0))
            else:
              weights.append(weight.flatten())
          feed_dict[self.contact_weights] = np.reshape(np.concatenate(weights, axis=0), (-1, 1))
        
        if not oneD_y_b is None and not predict and not self.oneD_labels.out_tensor is None:
          oneD_labels = []
          for ids, oneD_label in enumerate(oneD_y_b):
            oneD_labels.append(np.pad(oneD_label, ((0, max_n_res-oneD_label.shape[0]), (0,0)), 'constant'))
          feed_dict[self.oneD_labels] = np.concatenate(oneD_labels, axis=0)
        
        if not oneD_w_b is None and not predict and not self.oneD_weights.out_tensor is None:
          oneD_weights = []
          for ids, oneD_weight in enumerate(oneD_w_b):
            oneD_weights.append(np.pad(oneD_weight, ((0, max_n_res-oneD_weight.shape[0])), 'constant'))
          feed_dict[self.oneD_weights] = np.reshape(np.concatenate(oneD_weights, axis=0), (-1, 1))
          
        res_features = []
        res_flag_1D = []
        res_flag_2D = []
        for ids, seq_feat in enumerate(X_b):
          n_res = n_residues[ids]
          # Padding
          flag_1D = [1]*n_res + [0]*(max_n_res-n_res)
          if self.uppertri:
            flag_2D = [[0]*k + flag_1D[k:] for k in range(n_res)] + [[0]*max_n_res]*(max_n_res-n_res)
          else:
            flag_2D = [flag_1D]*n_res + [[0]*max_n_res]*(max_n_res-n_res)
          res_flag_1D.append(np.array(flag_1D))
          res_flag_2D.append(np.array(flag_2D))
          res_features.append(np.pad(seq_feat, ((0, max_n_res - n_res), (0, 0)), 'constant'))
        
        res_2D_features = []
        for ids, twoD_feat in enumerate(twoD_X_b):
          n_res = n_residues[ids]
          # Padding
          res_2D_features.append(np.pad(twoD_feat, ((0, max_n_res - n_res), (0, max_n_res - n_res), (0, 0)), 'constant'))
          
          
        if not self.res_features.out_tensor is None:
          feed_dict[self.res_features] = np.stack(res_features, axis=0)
        if not self.res_flag_1D.out_tensor is None:
          feed_dict[self.res_flag_1D] = np.stack(res_flag_1D, axis=0)
        if not self.res_flag_2D.out_tensor is None:
          feed_dict[self.res_flag_2D] = np.stack(res_flag_2D, axis=0)
        if not self.n_residues.out_tensor is None:
          feed_dict[self.n_residues] = np.array(n_residues)
        if not self.res_2D_features.out_tensor is None:
          feed_dict[self.res_2D_features] = np.stack(res_2D_features, axis=0)
        yield feed_dict

  def evaluate(self, dataset, metrics):
    """
    Evaluates the performance of this model on specified dataset.
    Parameters
    """
    # TODO: change this
    w_all = []
    y_all = []
    for _, _, y, w, _, _ in dataset.itersamples():
      w_all.append(np.sign(w))
      y_all.append(y)

    # Retrieve prediction label
    y_pred = self.predict_proba(dataset)
    if len(y_pred) == 2:
      y_pred = y_pred[1]
    # Mask all predictions and labels with valid index
    results = {}
    for metric in metrics:
      results[metric.name] = metric.compute_metric(y_all, y_pred, w=w_all)
    return results

  def predict_proba(self, dataset):
    # TODO: change this
    gen = self.default_generator(dataset)
    y_pred = []
    for feed_dict in gen:
      y_pred.append(self.session.run(self.outputs[-1], feed_dict=feed_dict))
    y_pred = np.concatenate(y_pred, 0)
    return y_pred
  
  def LossModule(self, n_input, in_layer):
    
    final_outputs_coordinates = Dense(out_channels=3, in_layers=[self.gather_out_layer], name='final_dense')
    distance_map = CoordinatesToDistanceMap(in_layers=[final_outputs_coordinates,
                                                       self.res_flag_2D,
                                                       self.n_residues], name='distance_map')
    
    self.contact_labels = Label(shape=(None, 1), name='labels_c')
    self.contact_weights = Weights(shape=(None, 1), name='weights_c')
    cost = WeightedL2Loss(in_layers=[distance_map, 
                                     self.contact_labels, 
                                     self.contact_weights], name='cost_r')
  
    self.add_output(distance_map)
    return 1, cost

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          **kwargs):
    # TODO: change this
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epoch, deterministic=deterministic),
        max_checkpoints_to_keep, checkpoint_interval, restore)

  def fit_generator(self,
                    feed_dict_generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False):
    """Train this model on data from a generator.

    Parameters
    ----------
    feed_dict_generator: generator
      this should generate batches, each represented as a dict that maps
      Layers to values.
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().

    Returns
    -------
    the average loss over the most recent checkpoint interval
    """
    # TODO: change this
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      loss = self.loss
      
      opt = self._get_tf('Optimizer')
      if not self.n_batches is None:
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value())) for tv in tvs]
      
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = opt.compute_gradients(self.loss.out_tensor, tvs)
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs) if gv[0] is not None]      
        train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
      
        self.session.run(tf.global_variables_initializer())
      else:
        train_step = opt.minimize(loss.out_tensor, global_step=self._get_tf("GlobalStep"))
      
      if checkpoint_interval > 0:
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      if restore:
        self.restore()
      avg_loss, n_averaged_batches = 0.0, 0.0
      n_samples = 0
      for feed_dict in self._create_feed_dicts(feed_dict_generator, True):
        n_samples += 1
        should_log = (self.tensorboard and
                      n_samples % self.tensorboard_log_frequency == 0)

        if not self.n_batches is None:
          fetches = accum_ops + [loss.out_tensor]
        else:
          fetches = [train_step, loss.out_tensor]
        if should_log:
          fetches.append(self._get_tf("summary_op"))
        fetched_values = self.session.run(fetches, feed_dict=feed_dict)
        if should_log:
          self._log_tensorboard(fetched_values[-1])
          avg_loss += fetched_values[-2]
        avg_loss += fetched_values[-1]
        n_averaged_batches += 1
        self.global_step += 1
        
        if not self.n_batches is None and self.global_step % self.n_batches == 0:
          self.session.run(train_step)
          self.session.run(zero_ops)
          
        if checkpoint_interval > 0 and self.global_step % checkpoint_interval == checkpoint_interval - 1:
          saver.save(self.session, self.save_file, global_step=self.global_step)
          avg_loss = float(avg_loss) / n_averaged_batches
          print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
          avg_loss, n_averaged_batches = 0.0, 0.0
      if n_averaged_batches > 0:
        avg_loss = float(avg_loss) / n_averaged_batches
      if checkpoint_interval > 0:
        if n_averaged_batches > 0:
          print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
        saver.save(self.session, self.save_file, global_step=self.global_step)
        time2 = time.time()
        print("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return avg_loss
  
class ConvNet3DStructure(ConvNet3DStructureBase):
  def __init__(self,
               **kwargs):
    super(ConvNet3DStructure, self).__init__(**kwargs)

  def Conv1DModule(self, n_input, in_layer):
    # TODO: change this
    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=32,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='global_conv_1'))
    
    in_layer = self.conv_1D_layers[-1]
    n_input = 32
    
    for i in range(2):
      # n_input = 32
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=17, name='Res1D_Module0_'+str(i)+'_')
    
    for i in range(3):
      # n_input = 32
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='Res1D_Module1_'+str(i)+'_')

    return n_input, in_layer
    
  def Conv2DModule(self, n_input, in_layer):
    # TODO: change this
    # n_input = 50
    n_input, in_layer = self.Res2DModule_c(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_down_')
    for i in range(60):
      # n_input = 50
      n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_Module'+str(i)+'_')
    return n_input, in_layer
