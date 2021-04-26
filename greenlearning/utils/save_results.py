import numpy as np
from .visualization import input_data_slice
from . import config

def save_results(model, Green_slice=1):
    """Save the Green's function evaluated at a grid in a csv file.
    If the spatial dimension is equal to 2, Green_slice indicates the slice to save the Green's function'
    """
    # Construct the input data depending on the dimension
    if model.dimension == 1:
        # Construct grid to evaluate the Green's function
        X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
        x_G_star = X_G.flatten()[:,None]
        y_G_star = Y_G.flatten()[:,None]
        input_data = np.concatenate((x_G_star,y_G_star),1)
        shape_Green = X_G.shape
        input_hom = model.x
    else:
        input_data, shape_Green, _, _ = input_data_slice(model, Green_slice)
        X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
        x1 = X_G.flatten()[:,None]
        x2 = Y_G.flatten()[:,None]
        input_hom = np.concatenate((x1,x2),1).astype(dtype=config.real(np))
    
    # Loop over the number of networks
    k = 0
    for i in range(model.n_output):
        for j in range(model.n_input):
            # Evaluate the Green's function
            G_pred_identifier = model.sess.run(model.G_network[i][j].evaluate(input_data))
            G_pred = G_pred_identifier.reshape(shape_Green)
    
            # Save Green's function into a csv file
            if model.dimension == 1:
                np.savetxt('%s/Green_%s_%s_%d.csv' % (model.path_csv, model.example_name, model.activation_name, k), G_pred, fmt='%.4e', delimiter=',')
            else:
                np.savetxt('%s/Green_%s_%s_%d-%d.csv' % (model.path_csv, model.example_name, model.activation_name, k, Green_slice), G_pred, fmt='%.4e', delimiter=',')
                
            k = k+1
        
        # Evaluate the homogeneous solution
        N_pred = model.sess.run(model.idn_N_pred[i].evaluate(input_hom))
        
        # Save homogeneous solution
        np.savetxt('%s/Hom_%s_%s_%d.csv' % (model.path_csv, model.example_name, model.activation_name, i), N_pred, fmt='%.4e', delimiter=',')
