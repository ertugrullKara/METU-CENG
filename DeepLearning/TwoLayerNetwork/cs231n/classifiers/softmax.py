import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[0]
  num_train = X.shape[1]
  
  for idx in xrange(num_train):
    scores = np.dot(W, X[:, idx])
    
    # To handle numerical instability.
    scores = scores - np.max(scores)
    
    loss += -scores[y[idx]] + np.log(np.sum(np.exp(scores)))
    
    # Convert to softmax function.
    scores = np.exp(scores)
    scores /= np.sum(scores)
    
    # To get rid of the extra if check.
    scores[y[idx]] -= 1
    for cls in xrange(num_classes):
      dW[cls, :] += scores[cls] * X[:, idx]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[0]
  num_train = X.shape[1]
  train_mask = np.arange(num_train)
  
  scores = np.dot(W, X)
  scores -= np.max(scores)
  scores = np.exp(scores)
  scores /= np.sum(scores, axis=0)
  loss = -np.sum(np.log(scores[y, train_mask]))
  
  scores[y, train_mask] -= 1
  dW = np.dot(scores, X.T)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
