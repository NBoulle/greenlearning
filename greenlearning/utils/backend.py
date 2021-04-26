"""Source: https://github.com/lululxvi/deepxde/blob/master/deepxde/backend.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_BACKEND = "tensorflow"
_VERSION = tf.__version__
_IS_TF_1 = _VERSION.startswith("1.")


if _IS_TF_1:
    print("Using TensorFlow 1 backend.\n")
    tf = tf.compat.v1
else:
    print("Using TensorFlow 2 backend.\n")
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def backend():
    """Returns the name and version of the current backend, e.g., ("tensorflow", 1.14.0).
    
    Returns:
        tuple: A ``tuple`` of the name and version of the backend GreenLearning is currently using.
        
    Example:
        
    .. code-block:: python
    
        gl.utils.backend.backend()
        >>> ("tensorflow", 1.15.0)
    """
    return _BACKEND, _VERSION


def is_tf_1():
    """Check the version of Tensorflow."""
    return _IS_TF_1