from .utils.backend import tf
from .utils import config

def initialize_weights(identifier):
    """Initialize the weights of the activation functions."""
    activation_weights = []
    
    # Create initial weights for rational activation function
    if identifier == 'rational':
        RP = tf.unstack(tf.Variable([1.1915, 1.5957, 0.5, 0.0218], dtype=config.real(tf)))
        RQ = tf.unstack(tf.Variable([2.383, 0.0, 1.0], dtype=config.real(tf)))
        activation_weights = [RP,RQ]
    
    return activation_weights

def rational(x, weights):
    """Define the rational activation function."""
    x = tf.math.divide(tf.math.polyval(weights[0],x), tf.math.polyval(weights[1],x))
    return x

def get(identifier, weights):
    """Return the activation function."""
    if isinstance(identifier, str):
        return {
                'elu': tf.nn.elu,
                'relu': tf.nn.relu,
                'selu': tf.nn.selu,
                'sigmoid': tf.nn.sigmoid,
                'sin': tf.sin,
                'tanh': tf.nn.tanh,
                'rational': lambda x:rational(x,weights),
                }[identifier]