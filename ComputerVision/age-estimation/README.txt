Backbone of the implementation is under "network.py"
class _Net in this file implements pyTorch network which wraps around pytorch.nn.Module class.

class Network handles the main parts. Such as; loading data and setting it up for pytorch, loading model, predicting outputs and initialising and training the model.


Driver code for the homework is under the2 ipython notebook. This notebook implements constructing network, finding and choosing hyper-parameters. Saving, loading models and doing tests and plotting outputs