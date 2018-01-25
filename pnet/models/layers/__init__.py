from conv_layers import ResidueEmbedding, Conv1DLayer, Conv2DLayer, \
    Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, Conv2DUp, \
    Conv1DAtrous, Conv2DAtrous, Conv2DBilinearUp, Conv2DASPP, BatchNorm
from diag_conv_layers import DiagConv2DAtrous, DiagConv2DLayer, DiagConv2DASPP
#from conv_layers_torch import TorchResidueEmbedding, TorchOuter, TorchContactMapGather, TorchResAdd
import deepchem
from deepchem.models.tensorgraph.layers import convert_to_layers
import tensorflow as tf
import numpy as np

class Expand_dim(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, dim=None, **kwargs):
    self.dim = dim
    super(Expand_dim, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.expand_dims(parent_tensor, self.dim)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class ToShape(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, n_filter, batch_size, **kwargs):
    self.n_filter = n_filter
    self.batch_size = batch_size
    super(ToShape, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    n_residues = inputs[0]
    shape = tf.reduce_max(n_residues)
    out_tensor = tf.stack([self.batch_size, shape, shape, self.n_filter], 0)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class ShapePool(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, n_filter=None, padding='SAME', **kwargs):
    self.n_filter = n_filter
    self.padding = padding
    super(ShapePool, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    shape_orig = inputs[0]
    if self.n_filter is None:
      n_filter = shape_orig[3]*2
    else:
      n_filter = self.n_filter
    if self.padding == 'VALID':
      out_tensor = tf.stack([shape_orig[0],
                             shape_orig[1]/2,
                             shape_orig[2]/2,
                             n_filter], 0)
    elif self.padding == 'SAME':
      out_tensor = tf.stack([shape_orig[0],
                             tf.to_int32(tf.ceil(tf.to_float(shape_orig[1])/2)),
                             tf.to_int32(tf.ceil(tf.to_float(shape_orig[2])/2)),
                             n_filter], 0)
    else:
      raise ValueError("padding not supported")
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class AminoAcidEmbedding(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self,
               pos_start=0,
               pos_end=25,
               **kwargs):
    self.pos_start = pos_start
    self.pos_end = pos_end
    super(AminoAcidEmbedding, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    input_features = in_layers[0].out_tensor
    amino_acid_features = in_layers[1].out_tensor

    i = tf.shape(input_features)[0]
    j = tf.shape(input_features)[1]
    embedding_length = tf.shape(amino_acid_features)[1]
    embedded_features = tf.reshape(tf.matmul(tf.reshape(input_features[:, :, self.pos_start:self.pos_end],
                                                        [i*j, self.pos_end - self.pos_start]),
                                             amino_acid_features),
                                   [i, j, embedding_length])

    out_tensor = tf.concat([embedded_features, input_features[:, :, self.pos_end:]], axis=2)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  
class AminoAcidPad(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self,
               embedding_length,
               **kwargs):
    self.embedding_length = embedding_length
    super(AminoAcidPad, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    AA_features = in_layers[0].out_tensor
    Pad_features = tf.Variable(tf.random_normal((4, self.embedding_length)))
    out_tensor = tf.concat([Pad_features[:1, :], AA_features[:20, :], Pad_features[1:, :], AA_features[20:, :]], axis=0)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class WeightedL2Loss(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(WeightedL2Loss, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    
    guess = in_layers[0].out_tensor
    label = in_layers[1].out_tensor
    weights = in_layers[2].out_tensor
    out_tensor = tf.reduce_sum(tf.square(guess - label), axis=1, keep_dims=True) * weights 
    out_tensor = tf.reduce_sum(out_tensor)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  
class AddThreshold(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(AddThreshold, self).__init__(in_layers, **kwargs)

  def build(self):
    self.threshold = tf.Variable(initial_value=np.log(0.8))
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    log_dist_pred = in_layers[0].out_tensor
    out_tensor = log_dist_pred - self.threshold
    
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  
class SigmoidLoss(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(SigmoidLoss, self).__init__(in_layers, **kwargs)
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    labels = in_layers[0].out_tensor[:, 1]
    logits = in_layers[1].out_tensor
    out_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class Sigmoid(deepchem.models.tensorgraph.layers.Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(Sigmoid, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    if len(in_layers) != 1:
      raise ValueError("Sigmoid must have a single input layer.")
    parent = in_layers[0].out_tensor
    out_tensor = tf.nn.sigmoid(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
