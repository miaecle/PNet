#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:28:44 2017
@author: zqwu
"""

import numpy as np
import tensorflow as tf
import pickle
import deepchem as dc
from rdkit import Chem
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, TensorWrapper, GraphConv, GraphPool, GraphGather, Add, \
    Reshape
from deepchem.models.tensorgraph.optimizers import Adam
from pnet.models.layers import BatchNorm, AminoAcidEmbedding, AminoAcidPad, \
    Conv1DLayer, Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, \
    WeightedL2Loss
from pnet.utils.amino_acids import AminoAcid_SMILES

def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.
  Turns y into vector of shape [n_samples, 2] (assuming binary labels).
  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.
  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)

class ConvNetContactMapBase(TensorGraph):
  """Base Class for Convolutional network contact map prediction"""
  def __init__(self,
               n_res_feat=56,
               embedding=True,
               embedding_length=100,
               max_n_res=1000,
               mode="classification",
               uppertri=True,
               learning_rate=1e-5,
               learning_rate_decay=0.96,
               weight1D=1.,
               **kwargs):
    """
    Parameters:
    -----------
    n_res_feat: int
      number of features for each residue
    embedding: bool, optional
      whether to transfer the first 25 features(one hot encoding of residue
      type) to variable embedding
    embedding_length: int, optional
      length of embedding
    max_n_res: int, optional
      maximum number of residues, used for padding
    mode: string, optional
      classification or regression
    uppertri: bool, optional
      build symmetry matrix(False) or upper-triangular matrix(True)
    """
    self.n_res_feat = n_res_feat
    self.embedding = embedding
    self.embedding_length = embedding_length
    self.max_n_res = max_n_res
    self.mode = mode
    self.uppertri = uppertri
    self.weight1D = weight1D
    super(ConvNetContactMapBase, self).__init__(**kwargs)
    
    self.conv_1D_layers = []
    self.conv_2D_layers = []
    self.res_layers = []
    
    global_step = self._get_tf("GlobalStep")
    self.learning_rate = tf.train.exponential_decay(learning_rate, 
                                                    global_step, 
                                                    10000, 
                                                    learning_rate_decay)
    self.set_optimizer(Adam(learning_rate=self.learning_rate,
                            beta1=0.9, 
                            beta2=0.999, 
                            epsilon=1e-7))
    self.module_count = 0
    self.build_graph()
    
    
  def build_graph(self):
    """ Build graph structure """
    with self._get_tf("Graph").as_default():
      self.res_features = Feature(shape=(self.batch_size, None, self.n_res_feat), name='res_features')
      self.training_placeholder = Feature(shape=(), dtype=tf.bool, name='training_placeholder')
      # Placeholder for valid index
      self.res_flag_1D = Feature(shape=(self.batch_size, None), dtype=tf.int32, name='flag_1D')
      self.res_flag_2D = Feature(shape=(self.batch_size, None, None), dtype=tf.int32, name='flag_2D')
      self.n_residues = Feature(shape=(self.batch_size,), dtype=tf.int32, name='n_res')
      self.amino_acid_features = self.amino_acid_embedding()
      
      n_input = self.n_res_feat
      in_layer = self.res_features
      if self.embedding:
        # Add embedding layer
        self.residues_embedding = AminoAcidEmbedding(
            pos_start=0,
            pos_end=25,
            embedding_length=self.embedding_length,
            in_layers=[in_layer, self.amino_acid_features], name='global_embedding')

        n_input = n_input - 25 + self.embedding_length
        in_layer = self.residues_embedding

      # User-defined structures
      n_input, self.conv1d_out_layer = self.Conv1DModule(n_input, in_layer)
      n_input, self.outer_out_layer = self.OuterModule(n_input, self.conv1d_out_layer)
      n_input, self.conv2d_out_layer = self.Conv2DModule(n_input, self.outer_out_layer)
      n_input, self.gather_out_layer = self.GatherModule(n_input, self.conv2d_out_layer)

      # 1D loss
      oneD_prediction = Dense(3, in_layers=[self.conv1d_out_layer], name='oneD_pred')
      self.oneD_prediction = Reshape((None, 3), in_layers=[oneD_prediction], name='oneD_reshape')
      self.add_output(self.oneD_prediction)
      self.oneD_labels = Label(shape=(None, 3), name='oneD_labels')
      self.oneD_weights = Weights(shape=(None, 1), name='oneD_weights')
      self.oneD_cost = WeightedL2Loss(in_layers=[self.oneD_prediction, 
                                                 self.oneD_labels, 
                                                 self.oneD_weights], name='oneD_cost')
      
      # Add loss layer
      if self.mode == "classification":
        softmax = SoftMax(in_layers=[self.gather_out_layer], name='softmax_pred')
        #self.add_output(softmax)
        self.contact_labels = Label(shape=(None, 2), name='labels_c')
        self.contact_weights = Weights(shape=(None, 1), name='weights_c')
        cost = SoftMaxCrossEntropy(in_layers=[self.contact_labels, self.gather_out_layer], name='cost_c')
        self.cost_balanced = WeightedError(in_layers=[cost, self.contact_weights], name='cost_balanced_c')
      elif self.mode == "regression":
        #self.add_output(self.gather_out_layer)
        self.contact_labels = Label(shape=(None, 1), name='labels_r')
        self.contact_weights = Weights(shape=(None, 1), name='weights_r')
        cost = WeightedL2Loss(in_layers=[self.gather_out_layer, 
                                         self.contact_labels, 
                                         self.contact_weights], name='cost_r')
        self.cost_balanced = cost
      
      self.all_cost = Add(weights=[1., self.weight1D], 
                          in_layers=[self.cost_balanced, self.oneD_cost], name='all_cost')
      self.set_loss(self.all_cost)
      return 

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True, 
                        **kwargs):
    """ Transform each batch into corresponding feed_dict """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, oneD_y_b, oneD_w_b) in dataset.iterbatches(
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
        
        if not self.res_features.out_tensor is None:
          feed_dict[self.res_features] = np.stack(res_features, axis=0)
        if not self.res_flag_1D.out_tensor is None:
          feed_dict[self.res_flag_1D] = np.stack(res_flag_1D, axis=0)
        if not self.res_flag_2D.out_tensor is None:
          feed_dict[self.res_flag_2D] = np.stack(res_flag_2D, axis=0)
        if not self.n_residues.out_tensor is None:
          feed_dict[self.n_residues] = np.array(n_residues)
        yield feed_dict

  def evaluate(self, dataset, metrics):
    """
    Evaluates the performance of this model on specified dataset.
    Parameters
    """
    w_all = []
    y_all = []
    for _, y, w, _, _ in dataset.itersamples():
      w_all.append(np.sign(w))
      y_all.append(y)

    # Retrieve prediction label
    y_pred = self.predict_proba(dataset)
    y_pred = self.rebuild_contact_map(y_pred, dataset)
    # Mask all predictions and labels with valid index
    results = {}
    for metric in metrics:
      results[metric.name] = metric.compute_metric(y_all, y_pred, w=w_all)
    return results

  def rebuild_contact_map(self, y_pred, dataset):
    n_residues = [len(seq) for seq in dataset._sequences]
    preds_out = []
    pos = 0
    for n_res in n_residues:
      out = np.zeros((n_res, n_res, 2))
      for i in range(n_res):
        out[i, i:, :] = y_pred[pos:pos+(n_res-i), :]
        out[i, i, :] = out[i, i, :]/2
        pos = pos + n_res - i
      out = out + np.transpose(out, axes=(1,0,2))
      preds_out.append(out)
    return preds_out
    
  def Conv1DModule(self, n_input, in_layer):
    raise NotImplementedError
  def OuterModule(self, n_input, in_layer):
    raise NotImplementedError
  def Conv2DModule(self, n_input, in_layer):
    raise NotImplementedError
  def GatherModule(self, n_input, in_layer):
    raise NotImplementedError
  
  def amino_acid_embedding(self, name=None):
    if name == None:
      name = 'AAEmbedding_'+str(self.module_count)+'_'
      self.module_count += 1
    feat = dc.feat.ConvMolFeaturizer()
    featurized_AA = [feat._featurize(Chem.MolFromSmiles(smile)) for smile in AminoAcid_SMILES]
    multiConvMol = ConvMol.agglomerate_mols(featurized_AA, max_deg=3)
    atom_features = TensorWrapper(tf.constant(multiConvMol.get_atom_features(), dtype=tf.float32), name=name+'atom_features')
    degree_slice = TensorWrapper(tf.constant(multiConvMol.deg_slice, dtype=tf.int32), name=name+'degree')
    membership = TensorWrapper(tf.constant(multiConvMol.membership, dtype=tf.int32), name=name+'membership')

    deg_adjs = []
    for i in range(0, 3):
      deg_adjs.append(TensorWrapper(tf.constant(multiConvMol.get_deg_adjacency_lists()[i+1], dtype=tf.int32), name=name+'deg_'+str(i)))
      
    gc1 = GraphConv(
        64,
        max_deg=3,
        activation_fn=tf.nn.relu,
        in_layers=[atom_features, degree_slice, membership] + deg_adjs, name=name+'gc1')
    batch_norm1 = BatchNorm(in_layers=[gc1, self.training_placeholder], name=name+'bn1')
    gp1 = GraphPool(max_degree=3, in_layers=[batch_norm1, degree_slice, membership] + deg_adjs, name=name+'gp1')
    gc2 = GraphConv(
        64,
        max_deg=3,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, degree_slice, membership] + deg_adjs, name=name+'gc2')
    batch_norm2 = BatchNorm(in_layers=[gc2, self.training_placeholder], name=name+'bn2')
    gp2 = GraphPool(max_degree=3, in_layers=[batch_norm2, degree_slice, membership] + deg_adjs, name=name+'gp2')
    dense = Dense(out_channels=self.embedding_length/2, activation_fn=tf.nn.relu, in_layers=[gp2], name=name+'dense1')
    batch_norm3 = BatchNorm(in_layers=[dense, self.training_placeholder], name=name+'bn3')
    readout = GraphGather(
        batch_size=21,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, degree_slice, membership] + deg_adjs, name=name+'gg')
    padding = AminoAcidPad(
        embedding_length=self.embedding_length,
        in_layers=[readout], name=name+'pad')
    return padding
  
  def Res1DModule_a(self, n_input, in_layer, size=3, name=None):
    if name == None:
      name = 'Res1D_up_'+str(self.module_count)+'_'
      self.module_count += 1
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name=name+'conv_b1'))
    in_layer_branch1 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input * 2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res1DModule_b(self, n_input, in_layer, size=3, name=None):
    if name == None:
      name = 'Res1D_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    in_layer_branch1 = in_layer
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//4,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input//4,
        n_size=size,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_a(self, n_input, in_layer, res_flag_2D=None, name=None):
    if name == None:
      name = 'Res2D_up_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
      
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_b1'))
    in_layer_branch1 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=3,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input * 2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_b(self, n_input, in_layer, res_flag_2D=None, name=None):
    if name == None:
      name = 'Res2D_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    in_layer_branch1 = in_layer
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//4,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input//4,
        n_size=3,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_c(self, n_input, in_layer, res_flag_2D=None, name=None):
    if name == None:
      name = 'Res2D_down_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_b1'))
    in_layer_branch1 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//4,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input//4,
        n_size=3,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//4,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input//2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer
  
class ConvNetContactMap(ConvNetContactMapBase):
  def __init__(self,
               **kwargs):
    super(ConvNetContactMap, self).__init__(**kwargs)

  def Conv1DModule(self, n_input, in_layer):

    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=64,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='global_conv_1'))
    
    in_layer = self.conv_1D_layers[-1]
    n_input = 64
    
    for i in range(3):
      # n_input = 64
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=17, name='Res1D_Module0_'+str(i)+'_')
    
    for i in range(5):
      # n_input = 64
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='Res1D_Module1_'+str(i)+'_')

    return n_input, in_layer
    
    
  def OuterModule(self, n_input, in_layer):
    # Add transform layer from 1D sequences to 2D sequences
    self.outer = Outer1DTo2DLayer(
        in_layers=[in_layer, self.n_residues, self.res_flag_2D], name='global_outer')
    n_input = n_input*3
    in_layer = self.outer
    return n_input, in_layer

  def Conv2DModule(self, n_input, in_layer):
    # n_input = 96
    n_input, in_layer = self.Res2DModule_c(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_down_')
    for i in range(20):
      # n_input = 96
      n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_Module'+str(i)+'_')
    return n_input, in_layer

  def GatherModule(self, n_input, in_layer):
    # Transform all channels of a single contact to predicitons of contact probability
    if self.mode == "classification":
      n_output = 2
    elif self.mode == "regression":
      n_output = 1
    self.gather_layer = ContactMapGather(
        n_input_feat=n_input,
        n_output=n_output,
        in_layers=[in_layer, self.res_flag_2D], name='global_gather')
    return n_output, self.gather_layer