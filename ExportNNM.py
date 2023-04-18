import ctypes
from ctypes import *
from enum import IntEnum
import struct
import tensorflow as tf
import numpy as np
import math

class THeader(Structure):
    _fields_ = [
     ('header', c_char * 4),
     ('ampType', c_char * 11),
     ('modelType', c_char),
     ('ampInfo', c_char * 32),
     ('addInfo1', c_char * 16),
     ('cabInfo', c_char * 32),
     ('addInfo2', c_char * 16)]
   
    def __init__(self):
        self.header = b'NNM1'

def BlockSize(n):
    return 2**math.ceil(math.log2(n))

def MPT(x, q = 8):
    l = x.size
    bs = BlockSize(q * l)
    hbs = int(bs/2)
    Hk = np.fft.fft(np.pad(x,bs))
    cn = np.fft.ifft(np.log(np.abs(Hk)))
    cn[1:hbs-1] *= 2
    cn[hbs+1:] = 0
    return np.fft.ifft(np.exp(np.fft.fft(cn)))[:l]        
        
def make_weights_array(model_to_save, output_scaling_gain=1.0):
    nnm = np.zeros(3432, c_float)

    for num_layer, layer in enumerate(model_to_save.layers):
      if isinstance(layer, tf.keras.layers.LSTM):
        wts = layer.get_weights()
        wu = np.concatenate(wts[:2]).transpose().flatten()
        b = wts[2].flatten()
        nnm[:1152] = np.concatenate((wu,b))

    for num_layer, layer in enumerate(model_to_save.layers):
      if isinstance(layer, tf.keras.layers.Dense):
        if layer.name == 'D2':
          d2 = np.concatenate(layer.get_weights(),None)
          nnm[1152+32:1152+32+17] = d2

    nnm[1152+32+17:1152+32+17+3] = np.array([0, 1, 1], c_float)

    for num_layer, layer in enumerate(model_to_save.layers):
      if isinstance(layer, tf.keras.layers.Dense):
        if layer.name == 'D1':
          nnm[1152+32+17:1152+32+17+3] = np.array([0, layer.get_weights()[0][0], output_scaling_gain], c_float)

    nnm[1152+32+17+3] = 1;

    for num_layer, layer in enumerate(model_to_save.layers):
      if isinstance(layer, tf.keras.layers.Conv1D):
        if layer.name == 'C1':
          c1 = MPT(layer.get_weights()[0].flatten())
          nnm[1152+32+17+3:1152+32+17+3+128] = np.flip(c1)

    nnm[1152+32+17+3+128] = 1;

    for num_layer, layer in enumerate(model_to_save.layers):
      if isinstance(layer, tf.keras.layers.Conv1D):
        if layer.name == 'C2':
          c2 = MPT(layer.get_weights()[0].flatten())
          nnm[1152+32+17+3+128:1152+32+17+3+128+2048] = c2

    return nnm

def export_model_to_nnm(filename, output_scaling_gain, dc_value, model_to_save, model_type = 0, amp_type='CLEAN', model_name='Untitled', cab_name='Untitled', add_info1='', add_info2=''):
    model_name = model_name.ljust(31)[:31]
    cab_name = cab_name.ljust(31)[:31]

    file = open(filename, 'wb')

    header = THeader()
    header.modelType = model_type
    header.ampType = amp_type[:11].encode('utf-8')
    header.ampInfo = model_name[:32].encode('utf-8')
    header.cabInfo = cab_name[:32].encode('utf-8')
    header.addInfo1 = add_info1[:16].encode('utf-8')
    header.addInfo2 = add_info2[:16].encode('utf-8')
    file.write(header)
  
    nnm = make_weights_array(model_to_save, output_scaling_gain)
    file.write(nnm)

    file.close()

def get_type_id(typeID, sampleRate = 44100):
  outID = typeID

  if sampleRate == 48000:
    outID += 10
  elif sampleRate == 88200:
    outID += 20
  elif sampleRate == 96000:
    outID += 30
  elif sampleRate == 176400:
    outID += 40
  elif sampleRate == 192000:
    outID += 50

  return(outID)