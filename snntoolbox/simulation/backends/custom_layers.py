import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer, Conv2D
from tensorflow.python.keras import activations

class NormAdd(Layer):
    
     def __init__(self, activation=None, **kwargs):
          self._initial_weights = kwargs.pop('weights', None)
          self.kwargs = kwargs
          self.activation = activations.get(activation)
          self.concat = keras.layers.Concatenate(axis=-1)
          super(NormAdd, self).__init__(**kwargs) 

     def build(self, input_shape): 
          n = len(input_shape)
          self.filters = input_shape[0][-1]

          # Weights
          if self._initial_weights is not None:
               weights_conv = self._initial_weights[0]
               bias_conv = self._initial_weights[1]
               self.b = self._initial_weights[2:]    
               self._initial_weights = None   
          else:    
               self.b = [None]*n
               for i in range(len(self.b)):
                    self.b[i] = self.add_weight(
                                   name="unshift"+str(i),
                                   shape = (self.filters,), 
                                   initializer = "zeros", 
                                   trainable = True
                                   )
                                   
               weights_conv = np.zeros([1, 1, n*self.filters, self.filters])
               for k in range(self.filters):
                    weights_conv[:, :, k::self.filters, k] = 1
               bias_conv = np.zeros(self.filters)
          
          self.conv = keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=1, 
                    weights=(weights_conv, bias_conv),
                    **self.kwargs
                    )    

          super(NormAdd, self).build(input_shape)

     def call(self, input_data):
          tensor = [None]*len(self.b)
          for i,image in enumerate(input_data):
               tensor[i] = image+self.b[i]
               
          out = self.concat(tensor)
          out = self.conv(out)
          
          if self.activation is not None:
               return self.activation(out)
          return out

     def compute_output_shape(self, input_shape): 
          return input_shape[0] + (self.filters,)

     def get_config(self):
        config = super().get_config().copy()
        config['weights'] = self.get_weights()
        config.update({
            'activation': self.activation,
            #'filters': self.filters
        })
        return config

     def set_weights(self, weights):
          conv_weights = self.conv.get_weights()
          if len(weights) == len(conv_weights):
               conv_weights = weights
               #print('--',self.name,' - Conv2D weights set.')
          elif len(weights) == len(conv_weights) + len(self.b):
               conv_weights = weights[:len(conv_weights)]
               self.b = [tf.convert_to_tensor(b, dtype=tf.float32) for b in weights[len(conv_weights):]]
               #print('--',self.name,' - Conv2D weights and input biases set.')
          else:
               print('<!!! - ',self.name,'> The weights provided do not match the layer shape. \n \
                    - Conv2D accepts list of len ',len(conv_weights),'. \n \
                    - Input biases accept list of len',len(self.b),'.\n \
                    Provided list of len ', len(weights),':\n        ',[tf.shape(w).numpy().tolist() for w in weights])
                    
          self.conv.set_weights([tf.convert_to_tensor(w, dtype=tf.float32) for w in conv_weights])

     def get_weights(self):
          return self.conv.get_weights()[:2]+self.b 


class NormReshape(Layer):
    
     def __init__(self, target_shape, **kwargs):
          self._initial_weights = kwargs.pop('weights', None)
          self.target_shape = target_shape
          self.resh = keras.layers.Reshape(self.target_shape, **kwargs)
          super(NormReshape, self).__init__(**kwargs) 

     def build(self, input_shape): 
          self.in_channels = input_shape[-1]

          # Weights
          if self._initial_weights is not None:
               self.lmbda = self._initial_weights[0]
               self.shift = self._initial_weights[1]
               self._initial_weights = None
          else:
               self.lmbda = self.add_weight(
                              name="lambda",
                              shape = (self.in_channels,),
                              initializer = "ones", trainable = False
                              )
               self.shift = self.add_weight(
                              name = "shift",
                              shape = (self.in_channels,),
                              initializer = "zeros", trainable = False
                              )
          super(NormReshape, self).build(input_shape)

     def call(self, input_data):
          out = input_data*(self.lmbda-self.shift)+self.shift   
          out = self.resh(out)
          return out

     def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape,
            'weights': self.get_weights()
        })
        return config

     def get_weights(self):
          return [self.lmbda, self.shift]
     
     def set_weights(self, weights):
          try:
               self.lmbda = weights[0]
               self.shift = weights[1]
          except ValueError:
               print('Weights need to be of shape: [',tf.shape(self.lmbda),',',tf.shape(self.shift),'].')


class NormConv2D(Conv2D):
     def __init__(self, **kwargs):
          self.initial_weights=kwargs.pop('weights', None)               
          super(NormConv2D, self).__init__(**kwargs) 

     def build(self, input_shape):
          super(NormConv2D, self).build(input_shape)
          # Normalization
          self.norm = [self.add_weight(
                         name="denom",
                         shape = (input_shape[-1],),
                         initializer = "ones", trainable = False
                         ),
                    self.add_weight(
                         name="shift",
                         shape = (input_shape[-1],),
                         initializer = "zeros", trainable = False
                         )
               ]
          if self.initial_weights is not None:
               self.set_weights(self.initial_weights)
               self._initial_weights = self.initial_weights
               del self.initial_weights

     def call(self, x):
          x = x*self.norm[0] + self.norm[1]
          return Conv2D.call(self, x)

     def set_weights(self,weights):
          if len(weights)==4:
               w = [tf.convert_to_tensor(u) for u in weights]
               super().set_weights(w)
          elif len(weights)==2:
               extra = tf.convert_to_tensor(self.norm)
               super().set_weights(list(weights)+[extra[0]]+[extra[1]])
          else:
               raise ValueError(
               'You called `set_weights(weights)` on layer "%s" '
               'with a weight list of length %s, but the layer was '
               'expecting 4 weights. Provided weights: %s...' %
               (self.name, len(weights), str(weights)[:50]))

     def get_weights(self):
          extra = tf.convert_to_tensor(self.norm)
          return [self.kernel, self.bias, extra[0], extra[1]]
