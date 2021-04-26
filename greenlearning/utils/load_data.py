import os
import numpy as np
import scipy.io
from . import config
from ..quadrature_weights import get_weights

def load_data(model, example_path, example_name):
    """Load the training dataset."""
    
    # Example name
    model.example_name = example_name
    
    # Set up paths to save the results
    try:
        os.mkdir(model.path_result)
    except:
        pass
    
    try:
        os.mkdir(model.path_csv)
    except:
        pass
    
    # Create training folder
    try:
        os.mkdir(model.path_training)
    except:
        pass
    
    # Activation function
    model.activation_name = model.G_network[0][0].activation_name
    
    # Name of the example
    model.example_name = example_name
    
    # Load the dataset
    model.data_idn = scipy.io.loadmat(example_path+"%s.mat" % model.example_name)
    
    # Get the training points x,y
    model.x = model.data_idn['X'].astype(dtype=config.real(np))
    model.y = model.data_idn['Y'].astype(dtype=config.real(np))
    
    # Get the spatial dimension
    model.dimension = model.x.shape[1]
    
    if model.G_network[0][0].layers[0] != 2*model.dimension:
        raise ValueError("First layer of G: %d does not match dimension 2d = %d." % (model.G_network[0][0].layers[0],2*model.dimension))
    
    # Construct quadrature weights
    quadrature_rule = "trapezoidal"
    if model.dimension > 1:
        quadrature_rule = "uniform"
    
    model.weights_x = get_weights(quadrature_rule, model.x)
    model.weights_y = np.reshape(get_weights(quadrature_rule, model.y), (-1,1,1))
    
     # Get the training data u and f
    model.u = model.data_idn['U'].astype(dtype=config.real(np))
    model.f = model.data_idn['F'].astype(dtype=config.real(np))
    
    # Reshape the training data to 3 dimensions
    if len(model.u.shape) == 2:
        model.u = np.reshape(model.u, model.u.shape+(1,))
    if len(model.f.shape) == 2:
        model.f = np.reshape(model.f, model.f.shape+(1,))
    
    # Check the network shape with respect to training data
    G_shape = (model.n_output, model.n_input)
    expected_shape = (model.u.shape[2], model.f.shape[2])
    
    if G_shape != expected_shape:
        raise ValueError("Shape of G: (%d,%d) and training data: (%d,%d) don't match." % (G_shape[0],G_shape[1],expected_shape[0],expected_shape[1]))
    
    # Load evaluation points and testing set
    model.U_hom = model.data_idn['U_hom'].astype(dtype=config.real(np))
    if len(model.U_hom.shape) == 1:
        model.U_hom = np.reshape(model.U_hom, model.U_hom.shape + (1,))
    
    model.x_G = model.data_idn['XG'].astype(dtype=config.real(np))
    model.y_G = model.data_idn['YG'].astype(dtype=config.real(np))
    
    try:
        model.ExactGreen = model.data_idn['ExactGreen'][0]
    except:
        model.ExactGreen = '0'