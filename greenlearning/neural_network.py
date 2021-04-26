from .utils.backend import tf
from . import activations
from .utils import config
import numpy as np

class NeuralNetwork:
    """Create a fully connected neural network with given number of layers and 
    activation function.

    Example:

    .. code-block:: python
    
        gl.NeuralNetwork([2] + [50] * 4 + [1], "rational")
    
    creates a rational neural network with 4 hidden layers of 50 neurons.
    """
    
    def __init__(self, layers, activation_name):
        self.layers = layers
        self.activation_name = activation_name
        
        # Initialize weights of the neural network
        self.initialize_NN()
        
    def initialize_NN(self):
        """Initialize the weights of the neural network."""
        
        # Create list to store the weights
        self.weights = []
        self.biases = []
        self.activation_weights = []
        
        # Loop over the number of layers
        for l in range(0,len(self.layers)-2):
            # Initialize weights and biases
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]]), dtype=config.real(tf))
            self.weights.append(W)
            self.biases.append(b)
            
            # Initialize the activation function weights
            self.activation_weights.append(activations.initialize_weights(self.activation_name))
            
        # Initialize last layer
        W = self.xavier_init(size=[self.layers[-2], self.layers[-1]])
        b = tf.Variable(tf.zeros([1,self.layers[-1]]), dtype=config.real(tf))
        self.weights.append(W)
        self.biases.append(b)
    
    def xavier_init(self, size):
        """Initializer for weights and biases."""
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=config.real(tf))    
    
    def evaluate(self, X):
        """Evaluate the neural network at the array X."""
        
        # Loop over the number of layers
        for l in range(0,len(self.layers)-2):
            W = self.weights[l]
            b = self.biases[l]
            X = tf.add(tf.matmul(X, W), b)
            
            # Add the activation function with corresponding weights
            X = activations.get(self.activation_name, self.activation_weights[l])(X)
            
        # Add the final layer
        W = self.weights[-1]
        b = self.biases[-1]
        X = tf.add(tf.matmul(X, W), b)
        return X
