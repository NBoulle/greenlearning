from .plotting import newfig, savefig, MathTextSciFormatter
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import config

def plot_results(model):
    """Plot the learned Green's function and homogeneous solution in a pdf."""
    
    # Is is a system of equation
    is_system = max(model.n_input,model.n_output) > 1
    
    # Choose the plotting function depending on the type of training data
    if model.dimension == 1 and not(is_system):
        plot_1d_results(model)
        
    elif model.dimension == 1 and is_system:
        plot_1d_systems(model)
        
    elif model.dimension > 1 and not(is_system):
        plot_2d_results(model)
        
    else:
        # Plot and save slices of the Green's matrix
        for i in range(1,5):
            plot_2d_systems(model, Green_slice=i)

def plot_1d_results(model):
    """Plot the learned Green's function and homogeneous solution in 1D."""
            
    # Construct grid to evaluate the Green's function
    X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
    x_G_star = X_G.flatten()[:,None]
    y_G_star = Y_G.flatten()[:,None]
    input_data = np.concatenate((x_G_star,y_G_star),1)
    
    # Evaluate the Green's function
    G_pred_identifier = model.sess.run(model.G_network[0][0].evaluate(input_data))
    G_pred = G_pred_identifier.reshape(X_G.shape)
    
    # Evaluate the homogeneous solution
    N_pred = model.sess.run(model.idn_N_pred[0].evaluate(model.x))
    
    # Evaluate exact Green's function
    try:
        G_expression = lambda x,y: eval(model.ExactGreen + '+ 0*x + 0*y')        
        Exact_idn = G_expression(X_G, Y_G)
    except:
        warnings.warn('Error in expression for the exact Green\'s function, assuming it is unknown.')
        Exact_idn = 0*X_G + 0*Y_G
    
    # Compute and print relative error
    Mean_Exact_Green = np.square(Exact_idn).mean()
    L2_norm_Green = np.mean(np.square(Exact_idn - G_pred))
    if Mean_Exact_Green == 0:
        output_function = np.sqrt(L2_norm_Green)
        print('Exact Green\'s function unknown, L2 norm = %g"'% output_function)
    else:
        output_function = np.sqrt(L2_norm_Green / Mean_Exact_Green)
        print('Relative error = %g' % output_function) 
    
    # Create the figure
    fig, ax = newfig(1.0, 1.5)
    ax.axis('off')
    gs = gridspec.GridSpec(2, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.6, hspace=0.6)
    
    # Plot exact Green's function if known
    ax = plt.subplot(gs[0, 0])
    if Mean_Exact_Green == 0:
        ax.text(0.5,0.5, "Unknown", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.set_xlim(np.min(model.x_G), np.max(model.x_G))
        ax.set_ylim(np.min(model.y_G), np.max(model.y_G))
    else:
        h = ax.imshow(Exact_idn, interpolation='lanczos', cmap='jet', 
                      extent=[np.min(model.x_G), np.max(model.x_G), np.min(model.y_G), np.max(model.y_G)],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$', rotation=0, labelpad=12)
    ax.set_title('Exact Green\'s function', fontsize = 10)
    
    # Plot the learned Green's function
    ax = plt.subplot(gs[0, 1])
    h = ax.imshow(G_pred, interpolation='lanczos', cmap='jet', 
                  extent=[np.min(model.x_G), np.max(model.x_G), np.min(model.y_G), np.max(model.y_G)],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$', rotation=0, labelpad=12)
    ax.set_title('Learned Green\'s function', fontsize = 10)
    
    # Plot training solutions u
    ax = plt.subplot(gs[1, 0])
    for i in range(model.u.shape[1]):
        ax.plot(model.x, model.u[:,i])
    divider = make_axes_locatable(ax)
    ax.set_xlim(np.min(model.x), np.max(model.x))
    ax.set_ylim(np.min(model.u), np.max(model.u))
    ax.set_xlabel('$x$')
    ax.set_title('Training solutions', fontsize = 10)

    # Plot the homogeneous solution
    ax = plt.subplot(gs[1, 1])
    ax.plot(model.x, model.U_hom, label='Exact')
    ax.plot(model.x, N_pred, dashes=[2, 2], label='Learned')
    divider = make_axes_locatable(ax)
    ax.set_xlim(np.min(model.x), np.max(model.x))
    # Determine the axis limit
    ymin = min([np.min(N_pred), np.min(model.U_hom)])
    ymax = max([np.max(N_pred), np.max(model.U_hom)])
    ax.set_ylim(ymin, ymax)
    if ymax - ymin < 1e-2:
        ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax.set_xlabel('$x$')
    ax.set_title('Homogeneous solution', fontsize = 10)
    ax.legend()
    
    # Save the figure
    savefig("%s/%s_%s" % (model.path_result, model.example_name, model.activation_name), crop=False)

def plot_1d_systems(model):
    """Plot the learned Green's functions and homogeneous solutions for a system of PDEs in 1D."""
    
    # Construct grid to evaluate the Green's function
    X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
    x_G_star = X_G.flatten()[:,None]
    y_G_star = Y_G.flatten()[:,None]
    
    # Create the figure
    n_plots = max(model.n_input+1,model.n_output)
    scaling = n_plots / 2
    fig, ax = newfig(1.0*scaling, 1.5)
    ax.axis('off')
    gs = gridspec.GridSpec(n_plots,n_plots)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.6*scaling, hspace=0.6*scaling)
    
    # Plot the Green's functions
    for i in range(model.n_output):
        for j in range(model.n_input):
            input_data = np.concatenate((x_G_star,y_G_star),1)
            G_pred_identifier = model.sess.run(model.G_network[i][j].evaluate(input_data))
            G_pred = G_pred_identifier.reshape(X_G.shape)
            ax = plt.subplot(gs[i, j])
            h = ax.imshow(G_pred, interpolation='lanczos', cmap='jet', 
                          extent=[np.min(model.x_G), np.max(model.x_G), np.min(model.y_G), np.max(model.y_G)],
                          origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$', rotation=0, labelpad=12)
            ax.set_title('$G_{%d,%d}$'%(i+1,j+1), fontsize = 10)
    
    # Plot the homogeneous solutions
    for i in range(model.n_output):
        N_pred = model.sess.run(model.idn_N_pred[i].evaluate(model.x))
        ax = plt.subplot(gs[i, model.n_input])
        ax.plot(model.x, model.U_hom[:,i], label='Exact')
        ax.plot(model.x, N_pred, dashes=[2, 2], label='Learned')
        divider = make_axes_locatable(ax)
        ymin = min([np.min(N_pred), np.min(model.U_hom[:,i])])
        ymax = max([np.max(N_pred), np.max(model.U_hom[:,i])])
        ax.set_xlim(np.min(model.x), np.max(model.x))
        ax.set_ylim(ymin, ymax)
        if ymax - ymin < 1e-2:
            ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        ax.set_xlabel('$x$')
        ax.set_title('Hom$_{%d}$'%(i+1), fontsize = 10)
        ax.legend()        
    
    # Save the figure
    savefig("%s/%s_%s" % (model.path_result, model.example_name,  model.activation_name), crop=False)

def plot_2d_results(model):
    """Plot the learned Green's function and homogeneous solution in 2D."""
    
    # Create the figure            
    fig, ax = newfig(1.5, 1.5)
    ax.axis('off')
    gs = gridspec.GridSpec(3, 3)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.7, hspace=0.7)
    
    # Loop to plot different slices of the Green's function
    for i in range(2):
        for j in range(2):
            # Evaluate the Green's function network
            input_data, shape_Green, axis_limit, axis_labels = input_data_slice(model, Green_slice=2*i+j+1)
            G_pred_identifier = model.sess.run(model.G_network[0][0].evaluate(input_data))
            G_pred = G_pred_identifier.reshape(shape_Green)
            ax = plt.subplot(gs[i, j])
            h = ax.imshow(G_pred, interpolation='lanczos', cmap='jet', 
                          extent=axis_limit,
                          origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1], rotation=0, labelpad=12)
            ax.set_title('$G$%s'%(axis_labels[2]), fontsize = 10)
    
    # Get axis limit
    x1min = np.min(model.x_G)
    x1max = np.max(model.x_G)
    x2min = np.min(model.y_G)
    x2max = np.max(model.y_G)
    
    # Exact Homogeneous solution
    ax = plt.subplot(gs[0, 2])
    X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
    x1 = X_G.flatten()[:,None]
    x2 = Y_G.flatten()[:,None]
    U_hom = np.transpose(model.U_hom.reshape((model.y_G.shape[0],model.x_G.shape[0])))
    h = ax.imshow(U_hom, interpolation='lanczos', cmap='jet', 
                  extent=[x1min,x1max,x2min,x2max],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$', rotation=0, labelpad=12)
    ax.set_title('Exact Hom', fontsize = 10)
    
    # Learned Homogeneous solution
    ax = plt.subplot(gs[1, 2])
    input_data = np.concatenate((x1,x2),1).astype(dtype=config.real(np))
    N_pred_identifier = model.sess.run(model.idn_N_pred[0].evaluate(input_data))
    N_pred = N_pred_identifier.reshape(X_G.shape)
    h = ax.imshow(N_pred, interpolation='lanczos', cmap='jet', 
                  extent=[x1min,x1max,x2min,x2max],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$', rotation=0, labelpad=12)
    ax.set_title('Learned Hom', fontsize = 10)
    
    # Save the figure
    savefig("%s/%s_%s" % (model.path_result, model.example_name, model.activation_name), crop=False)

def input_data_slice(model, Green_slice=1):
    """Return evaluation points for the selected slice of the Green's function.
    Green_slice=1 : G(:,:,y1,y2), where y1, y2 are points in the middle of the domain.
    Green_slice=2 : G(:,x2,:,y2).
    Green_slice=3 : G(x1,x2,:,:), where x1, x2 are points in the middle of the domain.
    Green_slice=4 : G(x1,:,y1,:).
    """
    
    # Define axis limit
    x1min = np.min(model.x_G) 
    x1max = np.max(model.x_G)
    x2min = np.min(model.y_G)
    x2max = np.max(model.y_G)
    x1_middle = 0.5*(x1max+x1min)
    x2_middle = 0.5*(x2max+x2min)
    
    # Compute the input data for the selected slice
    if Green_slice == 1:
        X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
        x1 = X_G.flatten()[:,None]
        x2 = Y_G.flatten()[:,None]
        y1 = x1_middle*np.ones(x1.shape)
        y2 = x2_middle*np.ones(x2.shape)
        axis_limit = [x1min,x1max,x2min,x2max]
        axis_labels = ['$x_1$','$x_2$','$(x_1,x_2,%.1f,%.1f)$'%(x1_middle,x2_middle)]
        
    elif Green_slice == 2:
        X_G, Y_G = np.meshgrid(model.x_G, model.x_G)
        x1 = X_G.flatten()[:,None]
        y1 = Y_G.flatten()[:,None]
        x2 = x2_middle*np.ones(x1.shape)
        y2 = x2_middle*np.ones(y1.shape)  
        axis_limit = [x1min,x1max,x1min,x1max]
        axis_labels = ['$x_1$','$y_1$','$(x_1,%.1f,y_1,%.1f)$'%(x2_middle,x2_middle)]
        
    elif Green_slice == 3:
        X_G, Y_G = np.meshgrid(model.y_G, model.y_G)
        x2 = X_G.flatten()[:,None]
        y2 = Y_G.flatten()[:,None]
        x1 = x1_middle*np.ones(x2.shape)
        y1 = x1_middle*np.ones(y2.shape)
        axis_limit = [x2min,x2max,x2min,x2max]
        axis_labels = ['$x_2$','$y_2$','$(%.1f,x_2,%.1f,y_2)$'%(x1_middle,x1_middle)]
        
    elif Green_slice == 4:
        X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
        y1 = X_G.flatten()[:,None]
        y2 = Y_G.flatten()[:,None]
        x1 = x1_middle*np.ones(y1.shape)
        x2 = x2_middle*np.ones(y2.shape)
        axis_limit = [x1min,x1max,x2min,x2max]
        axis_labels = ['$y_1$','$y_2$','$(%.1f,%.1f,y_1,y_2)$'%(x1_middle,x2_middle)]
        
    else:
        raise ValueError("Function not implemented for selected Green slice argument.")
    
    # Define input data and shape of the Green's function
    input_data = np.concatenate((x1,x2,y1,y2),1).astype(dtype=config.real(np))
    shape_Green = X_G.shape
    return input_data, shape_Green, axis_limit, axis_labels
    
def plot_2d_systems(model, Green_slice=1):
    """Plot the learned Green's functions and homogeneous solutions of a system of PDEs in 2D."""
    
    # Create evaluation points    
    input_data, shape_Green, axis_limit, axis_labels = input_data_slice(model, Green_slice)
    
    # Create the figure
    n_plots = max(model.n_input+1,model.n_output)
    scaling = n_plots / 2
    fig, ax = newfig(1.0*scaling, 1.5)
    ax.axis('off')
    gs = gridspec.GridSpec(n_plots,n_plots)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.6*scaling, hspace=0.6*scaling)

    # Plot the Green's function
    for i in range(model.n_output):
        for j in range(model.n_input):
            G_pred_identifier = model.sess.run(model.G_network[i][j].evaluate(input_data))
            G_pred = G_pred_identifier.reshape(shape_Green)
            ax = plt.subplot(gs[i, j])
            h = ax.imshow(G_pred, interpolation='lanczos', cmap='jet', 
                          extent=axis_limit,
                          origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1], rotation=0, labelpad=12)
            ax.set_title('$G_{%d,%d}$%s'%(i+1,j+1,axis_labels[2]), fontsize = 10)
    
    # Evaluation points for homogeneous solutions
    X_G, Y_G = np.meshgrid(model.x_G, model.y_G)
    x1 = X_G.flatten()[:,None]
    x2 = Y_G.flatten()[:,None]
    input_data = np.concatenate((x1,x2),1).astype(dtype=config.real(np))
    
    # Define axis limit
    x1min = np.min(model.x_G) 
    x1max = np.max(model.x_G)
    x2min = np.min(model.y_G)
    x2max = np.max(model.y_G)
    
    # Plot the homogeneous solution
    for i in range(model.n_output):
        N_pred_identifier = model.sess.run(model.idn_N_pred[i].evaluate(input_data))
        N_pred = N_pred_identifier.reshape(X_G.shape)
        ax = plt.subplot(gs[i, model.n_input])
        h = ax.imshow(N_pred, interpolation='lanczos', cmap='jet', 
                      extent=[x1min,x1max,x2min,x2max],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$', rotation=0, labelpad=12)
        ax.set_title('Hom$_{%d}$'%(i+1), fontsize = 10)
    
    # Save the figure
    savefig("%s/%s_%s-%d" % (model.path_result, model.example_name, model.activation_name, Green_slice), crop=False)