import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model

_loss="Softmax"
_num_affine_layer=1
_num_crp=1
_num_crcrp=0

def my_convnet_imp(X, model, y=None, reg=0.0):
  # Unpack
  loss = _loss
  num_affine_layer = _num_affine_layer
  num_crp = _num_crp
  num_crcrp = _num_crcrp
  W = []
  b = []
  crcrpflag = 0
  if num_crcrp > 0:
    crcrpflag=1
  conrelpool_layers = [num_crcrp if crcrpflag else num_crp][0]
  layers = conrelpool_layers + num_affine_layer
  for i in range(layers):
    W.append(model['W{}'.format(str(i))])
    b.append(model['b{}'.format(str(i))])
  N, C, H, _ = X.shape
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W[0].shape[2:]
  #assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  #assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  #assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': 0}
  pool_param = {'pool_height': 1, 'pool_width': 1, 'stride': 1}

  # Compute the forward pass
  #a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  #scores, cache2 = affine_forward(a1, W2, b2)
  layer_num=0
  a = []
  cache = []
  tmp=0
  for i in range(conrelpool_layers):
    if crcrpflag:
      a_tmp, cache_tmp = conv_relu_forward(X if len(a)==0 else a[-1], W[tmp], b[tmp], conv_param)
      a.append(a_tmp)
      cache.append(cache_tmp)
      layer_num+=1
      tmp+=1
    a_tmp, cache_tmp = conv_relu_forward(X if len(a)==0 else a[-1], W[tmp], b[tmp], conv_param)
    
    a.append(a_tmp)
    cache.append(cache_tmp)
    layer_num+=1
    tmp+=1
    
  for i in range(num_affine_layer):
    a_tmp, cache_tmp = affine_forward(a[-1], W[tmp], b[tmp])
    cache.append(cache_tmp)
    a.append(a_tmp)
    tmp+=1
    
  scores = a[-1]
    
  if y is None:
    return scores
  # Compute the backward pass
  if loss.lower() == "softmax":
    data_loss, dscores = softmax_loss(scores, y)
  elif loss.lower() == "svm":
    data_loss, dscores = svm_loss(scores, y)
  elif loss.lower() == "mse":
    data_loss, dscores = mse_loss(scores, y)

  # Compute the gradients using a backward pass
  #da1, dW2, db2 = affine_backward(dscores, cache2)
  #dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)
  
  dW = [0]*len(W)
  db = [0]*len(b)
  tmp=-1
  for i in range(num_affine_layer):
    da_tmp, dW_tmp, db_tmp = affine_backward(dscores, cache[tmp])
    dscores = da_tmp
    dW[i]=dW_tmp
    db[i]=db_tmp
    tmp-=1
  i=0
  while i < layer_num:
    if crcrpflag:
      da_tmp, dW_tmp, db_tmp = conv_relu_backward(dscores, cache[tmp])
      dscores = da_tmp
      dW[num_affine_layer+i]=dW_tmp
      db[num_affine_layer+i]=db_tmp
      i += 1
      tmp-=1
    da_tmp, dW_tmp, db_tmp = conv_relu_backward(dscores, cache[tmp])
    dscores = da_tmp
    dW[num_affine_layer+i]=dW_tmp
    db[num_affine_layer+i]=db_tmp
    i += 1
    tmp-=1
    
  # Add regularization
  dW.reverse()
  db.reverse()
  for i in range(len(W)):
    dW[i]+=reg*W[i]
  reg_loss = 0.5 * reg * sum(np.sum(w * w) for w in W)

  loss = data_loss + reg_loss
  grads = {}
  for i in range(conrelpool_layers+num_affine_layer):
      grads['W'+str(i)] = dW[i]
      grads['b'+str(i)] = db[i]
  
  return loss, grads


def init_my_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=[32], filter_size=5, loss="softmax",
                   num_crp=1, num_affine_layer=1, num_crcrp=0):
  """
  num_crcrp overrides num_crp.
  """
  C, H, W = input_shape
  prev = 0
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
  
  CC = C
  model = {}
  global _loss, _num_affine_layer, _num_crp, _num_crcrp
  _loss = loss
  _num_affine_layer = num_affine_layer+1
  _num_crp = num_crp
  _num_crcrp = num_crcrp
  if num_crcrp > 0:
    tmp=0
    for i in range(num_crcrp):
      model['W{}'.format(str(tmp))] = weight_scale * np.random.randn(num_filters[tmp], CC, 1, filter_size)
      model['b{}'.format(str(tmp))] = bias_scale * np.random.randn(num_filters[tmp])
      tmp += 1
      model['W{}'.format(str(tmp))] = weight_scale * np.random.randn(num_filters[tmp], CC, 1, filter_size)
      model['b{}'.format(str(tmp))] = bias_scale * np.random.randn(num_filters[tmp])
      tmp += 1
      CC = num_filters[i]
    prev = num_crcrp
  else:
    for i in range(num_crp):
      model['W{}'.format(str(i))] = weight_scale * np.random.randn(num_filters[i], CC, 1, filter_size)
      model['b{}'.format(str(i))] = bias_scale * np.random.randn(num_filters[i])
      CC = num_filters[i]
    prev = num_crp

  num_affine = [57088]
  for i in range(1, num_affine_layer+1):
    num_affine += [100]
  num_affine += [num_classes]
  #print num_affine
  for i in range(num_affine_layer+1):
    model['W{}'.format(str(prev+i))] = weight_scale * np.random.randn(num_affine[i], num_affine[i+1])
    model['b{}'.format(str(prev+i))] = bias_scale * np.random.randn(num_affine[i+1])
  #model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  #model['b1'] = bias_scale * np.random.randn(num_filters)
  #model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  #model['b2'] = bias_scale * np.random.randn(num_classes)
  return model