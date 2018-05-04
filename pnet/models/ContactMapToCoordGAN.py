# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
import time
import threading

import deepchem as dc
import tensorflow as tf
from deepchem.models import GAN

from rdkit import Chem
from deepchem.models.tensorgraph.layers import Input, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, TensorWrapper, GraphConv, GraphPool, GraphGather, Add, \
    Reshape, Squeeze, ReduceMean
    
from deepchem.models.tensorgraph.optimizers import Adam
from pnet.models.layers import BatchNorm, ResidueEmbedding, Conv1DLayer, \
    Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, \
    Conv2DUp, Expand_dim, ShapePool, ToShape, Conv1DAtrous, Conv2DAtrous, \
    Conv2DASPP, CoordinatesToDistanceMap, Condense, SpatialAttention, \
    CoordinateScale
    
from pnet.utils.amino_acids import AminoAcid_SMILES
from pnet.models.conv_net_contact_map import ConvNetContactMapBase, to_one_hot, from_one_hot

class ContactMapToCoordGAN(GAN):

  def __init__(self, n_conditional_input=1, n_noise_input=10, filter_size=3, **kwargs):
    
    self.conv_1D_layers = []
    self.conv_2D_layers = []
    self.res_layers = []
    self.module_count = 0
    
    # Conditional Input is the contact probability matrix
    self.n_conditional_input = n_conditional_input
    self.n_noise_input = n_noise_input
    self.filter_size = filter_size
    super(ContactMapToCoordGAN, self).__init__(**kwargs)
    
  def get_noise_input_shape(self):
    return (self.batch_size, None, None, self.n_noise_input)

  def get_data_input_shapes(self):
    return [(self.batch_size, None, 3)]

  def get_conditional_input_shapes(self):
    return [(self.batch_size, None, None, self.n_conditional_input)]


  def create_generator(self, noise_input, conditional_inputs):
    # Extra Inputs
    self.training_placeholder = Feature(shape=(), dtype=tf.bool, name='training_placeholder')
    # Placeholder for valid index
    self.res_flag_1D = Feature(shape=(self.batch_size, None), dtype=tf.int32, name='flag_1D')
    self.res_flag_2D = Feature(shape=(self.batch_size, None, None), dtype=tf.int32, name='flag_2D')
    self.n_residues = Feature(shape=(self.batch_size,), dtype=tf.int32, name='n_res')
    
    
    in_layer = Concat(axis=3, in_layers=[conditional_inputs[0], noise_input])
    n_input = self.n_conditional_input + self.n_noise_input
    
    # User-defined structures
    n_input, self.conv2d_out_layer = self.gen_atrous_structure(n_input, in_layer)
    n_input, self.condense_layer = self.gen_condense_module(n_input, self.conv2d_out_layer)   
    n_input, self.conv1d_out_layer = self.gen_conv1D_module(n_input, self.condense_layer)
    assert n_input == 3
    
    return [self.conv1d_out_layer]

  def gen_atrous_structure(self, n_input, in_layer):
    n_pool_layers = 2
    self.shortcut_layers = []
    self.encode_pool_layers = []
    self.decode_up_layers = []
    self.res_flags = [self.res_flag_2D]
    self.shortcut_shapes = [ToShape(16, self.batch_size, in_layers=[self.n_residues], name='gen_global_to_shape')]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=16,
        n_size=7,
        in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='gen_global_conv2d_0'))
    
    # All layers will be of shape: (batch_size, None, None, n_input)
    in_layer = self.conv_2D_layers[-1]
    n_input = 16
    
    res_flag_2D = self.res_flag_2D
    for i in range(n_pool_layers):
      for j in range(1):
        # n_input = 16
        n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Encoding_Module_'+str(i)+'_Submodule_'+str(j)+'_')

      self.shortcut_layers.append(in_layer)

      self.encode_pool_layers.append(Conv2DPool(
          n_size=2,
          in_layers=[in_layer], name='gen_global_pool_'+str(i)))
      in_layer = self.encode_pool_layers[-1]

      flag_2D = Expand_dim(3, in_layers=[res_flag_2D], name='gen_global_oldflag_'+str(i))
      new_flag_2D = Conv2DPool(n_size=2, in_layers=[flag_2D], name='gen_global_newflag_'+str(i))
      res_flag_2D = Squeeze(squeeze_dims=3, in_layers=[new_flag_2D], name='gen_global_flag_'+str(i))
      self.res_flags.append(res_flag_2D)
      self.shortcut_shapes.append(ShapePool(n_filter=n_input, in_layers=[self.shortcut_shapes[-1]], name='gen_global_shape_pool_'+str(i)))
      
    # n_input = 16
    n_input, in_layer = self.Res2DAtrousModule_b(n_input, in_layer, rate=2, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Inter_Module_'+str(i)+'_Rate_'+str(2)+'_')
    #n_input, in_layer = self.Res2DAtrousModule_b(n_input, in_layer, rate=4, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Inter_Module_'+str(2)+'_Rate_'+str(4)+'_')
    #n_input, in_layer = self.Res2DAtrousModule_b(n_input, in_layer, rate=8, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Inter_Module_'+str(3)+'_Rate_'+str(8)+'_')
    
    in_layer = self.conv_2D_layers[-1]
    
    for j in range(n_pool_layers):
      res_flag_2D = self.res_flags[-(j+2)]
      out_shape = self.shortcut_shapes[-(j+2)]
      self.decode_up_layers.append(Conv2DUp(
          n_input_feat=n_input,
          n_output_feat=n_input,
          n_size=2,
          in_layers=[in_layer, out_shape, res_flag_2D],
          name='gen_global_upconv_'+str(j)))
      
      # n_input = 32
      in_layer = Concat(axis=3, in_layers=[self.decode_up_layers[-1],
          self.shortcut_layers[-(j+1)]], name='gen_global_upconcat_'+str(j))
      
      n_input = n_input * 2
      n_input, in_layer = self.Res2DModule_c(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Decoding_Module_'+str(j)+'_Down_')
      for i in range(1):
        # n_input = 16
        n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='gen_Res2D_Decoding_Module_'+str(j)+'_Submodule_'+str(i)+'_')
    # n_input = 32
    n_input, in_layer = self.Res2DModule_a(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='gen_Res2D_UpModule_')
    return n_input, in_layer

  def gen_condense_module(self, n_input, in_layer):
    out_layer = Condense(in_layers=[in_layer, self.conditional_inputs[0]], name='gen_condense')
    # n_input = 64
    n_input = n_input * 2
    return n_input, out_layer
  
  def gen_conv1D_module(self, n_input, in_layer):
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=16,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='gen_global_conv1d_0'))
    
    n_input = 16
    in_layer = self.conv_1D_layers[-1]
    
    for i in range(1):
      # n_input = 16
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=17, name='gen_Res1D_LF_Module0_'+str(i)+'_')
    for i in range(1):
      # n_input = 16
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='gen_Res1D_Module1_'+str(i)+'_')
    # n_input = 32
    in_layer = SpatialAttention(in_layers=[in_layer, self.conditional_inputs[0], self.n_residues], name='gen_spatial_attention')
    n_input = n_input * 2
    for i in range(1):
      # n_input = 32
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='gen_Res1D_Module2_'+str(i)+'_')
    unnormalized_coordinates = Dense(3, in_layers=[in_layer], name='gen_generate_coordinates')
    #output_coordinates = CoordinateScale(in_layers=[unnormalized_coordinates], name='gen_output_scale')
    return 3, unnormalized_coordinates
  
  
  def create_discriminator(self, data_inputs, conditional_inputs):
    
    coords = data_inputs[0]
    dis_map = CoordinatesToDistanceMap(in_layers=[coords, self.n_residues])
    contact_prob = conditional_inputs[0]
            
    n_input, out_layer = self.dis_conv2d_module(1, contact_prob)
    n_input, out_layer = self.dis_condense_module(n_input, out_layer)   
    
    diff_2D = self.dis_diff_2D(dis_map, contact_prob)
    diff_1D = self.dis_diff_1D(n_input, out_layer, coords)
    
    in_layer = Concat(in_layers=[diff_2D, diff_1D], axis=1)
    dense = Dense(1, in_layers=[in_layer], activation_fn=tf.sigmoid)
    return dense  
  
  def dis_diff_1D(self, n_input, in_layer, coords):
    
    out_layer = Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=8,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='dis_global_conv1d_0')
        
    in_layer = Concat(in_layers=[out_layer, coords], axis=2)
    
    in_layer = Conv1DLayer(
        n_input_feat=8+3,
        n_output_feat=16,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='dis_global_conv1d_1')
    n_input = 16
    
    for i in range(3):
      # n_input = 16
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='dis_Res1D_Module1_'+str(i)+'_')
    scores = Dense(1, in_layers=[in_layer], name='dis_score_1D')
    final_score = ReduceMean(in_layers=[scores], axis=(1,), name='dis_final_score_1D')
    return final_score

  def dis_diff_2D(self, dis_map, contact_prob):
    
    in_layer = Concat(in_layers=[dis_map, contact_prob], axis=3)
    
    in_layer = Conv2DLayer(
        n_input_feat=2,
        n_output_feat=16,
        n_size=7,
        in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='dis_global_conv2d_1')
    n_input = 16
    
    for i in range(3):
      # n_input = 16
      n_input, in_layer = self.Res2DModule_b(n_input, in_layer, name='dis_Res1D_Module2_'+str(i)+'_')
    scores = Dense(1, in_layers=[in_layer], name='dis_score_2D')
    final_score = ReduceMean(in_layers=[scores], axis=(1, 2), name='dis_final_score_2D')
    return final_score
  
    
  def dis_conv2d_module(self, n_input, in_layer):
    in_layer = Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=16,
        n_size=7,
        in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='dis_global_conv2d_0')
    
    n_input = 16
    
    res_flag_2D = self.res_flag_2D
    for j in range(5):
      # n_input = 16
      n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='dis_Res2D_Module_'+str(j)+'_')

    # n_input = 32
    n_input, in_layer = self.Res2DModule_a(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='dis_Res2D_UpModule_')
    return n_input, in_layer

  def dis_condense_module(self, n_input, in_layer):
    out_layer = Condense(in_layers=[in_layer, self.conditional_inputs[0]], name='dis_condense')
    # n_input = 64
    n_input = n_input * 2
    return n_input, out_layer
  
  def Res1DAtrousModule_b(self, n_input, in_layer, rate=2, size=3, name=None):
    if name == None:
      name = 'Res1DAtrous_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    in_layer_branch1 = in_layer
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DAtrousModule_b(self, n_input, in_layer, rate=2, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'Res2DAtrous_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    in_layer_branch1 = in_layer
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer
  
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
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]    

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a4'))
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
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res1DModule_c(self, n_input, in_layer, size=3, name=None):
    if name == None:
      name = 'Res1D_down_'+str(self.module_count)+'_'
      self.module_count += 1
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
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
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]    

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input // 2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_a(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
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
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input * 2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_b(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'Res2D_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    in_layer_branch1 = in_layer
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def Res2DModule_c(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
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
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input//2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

model = ContactMapToCoordGAN(batch_size=2)
model.build()