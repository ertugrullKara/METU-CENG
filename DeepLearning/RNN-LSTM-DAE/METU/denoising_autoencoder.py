import numpy as np
import matplotlib.pyplot as plt

identity = lambda x: x

class DenoisingAutoencoder(object):
    """
    Denoising autoencoder.
    """
    def sigmoid(self, x):
      return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
      sig = self.sigmoid(x)
      return sig*(1-sig)

    def ac_func(self, x, function_name = 'SIGMOID'):
        # Implement your activation function here
      fname_upper = function_name.upper()
      if fname_upper =='SIGMOID':
        return self.sigmoid(x)
      else:
        raise fname_upper + " Not implemented Yet"

    def ac_func_deriv(self, x, function_name = 'SIGMOID'):
      # Implement the derivative of your activation function here
      fname_upper = function_name.upper()
      if fname_upper == 'SIGMOID':
        return self.sigmoid_deriv(x)
      else:
        raise fname_upper + " Not implemented Yet"

    def __init__(self, layer_units, weights=None):
        self.weights = weights
        self.layer_units = layer_units

    def init_weights(self, seed=0):
        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        # Note layer_units[2] = layer_units[0]
        layer_units = self.layer_units
        n_layers = len(layer_units)
        assert n_layers == 3

        np.random.seed(seed)

        # Initialize parameters randomly based on layer sizes
        r  = np.sqrt(6) / np.sqrt(layer_units[1] + layer_units[0])
        # We'll choose weights uniformly from the interval [-r, r)
        weights = {}
        weights['W0'] = np.random.random((layer_units[0], layer_units[1])) * 2.0 * r - r
        weights['W1'] = np.random.random((layer_units[1], layer_units[2])) * 2.0 * r - r
        weights['b0'] = np.zeros(layer_units[1])
        weights['b1'] = np.zeros(layer_units[2])

        self.weights = weights

        return self.weights

    def predict(self, X_noisy, reg=3e-3, activation_function='sigmoid'):
      weights = self.weights

      # Weight parameters
      W0 = weights['W0']
      b0 = weights['b0']
      W1 = weights['W1']
      b1 = weights['b1']

      # TODO: Implement forward pass here. It should be the same forward pass that you implemented in the loss function
      l0 = X_noisy.dot(W0) + b0
      l0_ac = self.ac_func(l0, activation_function)
      l1 = l0_ac.dot(W1) + b1
      l1_ac = self.ac_func(l1, activation_function)

      scores = l1_ac
      
      return scores

    def loss(self, X_noisy, X, reg=3e-3, activation_function='sigmoid'):
      weights = self.weights

      # Weighting parameters
      W0 = weights['W0']
      b0 = weights['b0']
      W1 = weights['W1']
      b1 = weights['b1']

      scores = None
      #############################################################################
      # TODO: Perform the forward pass, computing the  scores for the input. 	    #
      # Store the result in the scores variable, which should be an array of      #
      # shape (N, N).                                                             #
      #############################################################################
      l0 = X_noisy.dot(W0) + b0
      l0_ac = self.ac_func(l0, activation_function)
      l1 = l0_ac.dot(W1) + b1
      l1_ac = self.ac_func(l1, activation_function)

      scores = l1_ac
      #############################################################################
      #                              END OF YOUR CODE                             #
      #############################################################################

      #############################################################################
      # TODO: Compute the loss. This should include 				    #
      #             (i) the data loss (square error loss),			    #
      #             (ii) L2 regularization for W1 and W2, and    		    #
      # Store the result in the variable loss, which should be a scalar.          #
      # (Don't forget to investigate the effect of L2 loss)                       #
      #############################################################################
      scores = scores - X
      # print X.shape
      loss = 0.5 * np.sum(np.square(scores), axis=1).mean()
      loss += 0.5 * reg * (np.sum(W0**2) + np.sum(W1**2))
      #############################################################################
      #                              END OF YOUR CODE                             #
      #############################################################################

      grads = {}
      #############################################################################
      # TODO: Compute the backward pass, computing the derivatives of the weights #
      # and biases. Store the results in the grads dictionary. For example,       #
      # grads['W1'] should store the gradient on W1, and be a matrix of same size #
      #############################################################################
      grad = scores*self.ac_func_deriv(l1, activation_function)
      grads["W1"] = l0_ac.T.dot(grad)/X_noisy.shape[0] + reg*W1
      grads["b1"] = grad.sum(axis=0)/X_noisy.shape[0]
      
      grad = grad.dot(W1.T)*self.ac_func_deriv(l0, activation_function)
      grads["W0"] = X_noisy.T.dot(grad)/X_noisy.shape[0] + reg*W0
      grads["b0"] = grad.sum(axis=0)/X_noisy.shape[0]

      #############################################################################
      #                              END OF YOUR CODE                             #
      #############################################################################

      return loss, grads

    def train_with_SGD(self, X, noise=identity,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=3e-3, num_iters=100,
            batchsize=128, momentum='classic', mu=0.9, verbose=False, 
            activation_function='sigmoid'):	

        num_train = X.shape[0]

        loss_history = []

        layer_units = self.layer_units
        n_layers = len(layer_units)
        velocity = {}
        velocity['W0'] = np.zeros((layer_units[0], layer_units[1]))
        velocity['W1'] = np.zeros((layer_units[1], layer_units[2]))
        velocity['b0'] = np.zeros(layer_units[1])
        velocity['b1'] = np.zeros(layer_units[2])

        for it in xrange(num_iters):

              batch_indicies = np.random.choice(num_train, batchsize, replace = False)
              X_batch = X[batch_indicies]

              # Compute loss and gradients
              noisy_X_batch = noise(X_batch)
              loss, grads = self.loss(noisy_X_batch, X_batch, reg, activation_function=activation_function)
              loss_history.append(loss)

              #########################################################################
              # TODO: Use the gradients in the grads dictionary to update the         #
              # parameters of the network (stored in the dictionary self.params)      #
              # using gradient descent.                                               #
              #########################################################################


              # You can start and test your implementation without momentum. After 
              # making sure that it works, you can add momentum
              for key in velocity.keys():
                velocity[key] = velocity[key]*mu - learning_rate*grads[key]
                self.weights[key] += velocity[key]
              #########################################################################
              #                             END OF YOUR CODE                          #
              #########################################################################

              if verbose and it % 10 == 0:
                    print 'SGD: iteration %d / %d: loss %f' % (it, num_iters, loss)

              # Every 5 iterations.
              if it % 5 == 0:
                    # Decay learning rate
                    learning_rate *= learning_rate_decay

        return { 'loss_history': loss_history, }
