import numpy as np
from .utils.backend import tf
from .utils.print_weights import print_weights
from .utils.load_data import load_data
from .utils.visualization import plot_results
from .utils.save_results import save_results
from .utils.tf_session import open_tf_session
from .loss_function import loss_function
from .utils.external_optimizer import ScipyOptimizerInterface

class Model:
    """Create a model to learn Green's function from input-output data with deep
    learning.
    
    Example:
    
    .. code-block:: python
        
         # Construct neural networks for G and homogeneous solution
        G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (2,2))
        U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (2,))
        
        # Define the model
        model = gl.Model(G_network, U_hom_network)
        
        # Train the model on the selected dataset in the path "examples/datasets/"
        model.train("examples/datasets/", "ODE_system")
        
        # Plot the results
        model.plot_results()
        
        # Close the session
        model.sess.close()
    
    """
    
    def __init__(self, G_network, U_hom_network):
        """Initialize the model."""
        
        # Paths to save the results
        self.path_csv = "results_csv"
        self.path_result = "results"
        self.path_training = "training"
        
        # Number of epochs for Adam and L-BFGS optimizers
        self.epochs_adam = 10**3
        self.epochs_lbgs = 5*10**4
        
        # Number of input and output data
        self.n_input = len(G_network[0])
        self.n_output = len(G_network)
        
        # Check the networks shape
        if self.n_output != len(U_hom_network):
            raise ValueError("Output shape of the networks must match: G = (%d,%d), U_hom = (%d,)."%(self.n_output,self.n_input,len(U_hom_network)))
        
        # Neural networks for Green's function and homogeneous solution
        self.G_network = G_network
        self.idn_N_pred = U_hom_network
        
        if self.G_network[0][0].layers[0] != 2*self.idn_N_pred[0].layers[0]:
            raise ValueError("First layer of G: %d must be twice larger than first layer of homogeneous network: %d." % (self.G_network[0][0].layers[0],self.idn_N_pred[0].layers[0]))
            
        # Initialize the optimizers
        self.init_optimizer()
        
        # Create tensorflow session
        self.sess = open_tf_session()
    
    def init_optimizer(self):
        """Initialize the variables and optimizers."""
                
        # Define loss function        
        self.idn_loss = loss_function(self.G_network, self.idn_N_pred)
        
        # Define Adam optimizaer
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.idn_loss.outputs, var_list=tf.trainable_variables())
        
        # Define L-BFGS-B optimizer
        self.idn_u_optimizer = ScipyOptimizerInterface(self.idn_loss.outputs,
                                var_list = tf.trainable_variables(),
                                method = 'L-BFGS-B',
                                options = {'maxiter': self.epochs_lbgs,
                                           'maxfun': 10**5,
                                           'iprint':10,
                                           'iprint':10,
                                           'maxcor': 50,
                                           'maxls': 50,
                                           'ftol': 1e-20,
                                           'gtol': 1.0*np.finfo(float).eps})
        
    def train(self, example_path, example_name):
        """Train the Green's function and homogeneous solution networks."""
        
        # Choose the data set and load it
        load_data(self, example_path, example_name)
        
        # Initialize the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # Create the feed dictionnary        
        tf_dict = self.idn_loss.feed_dict(self.x, self.y, self.f, self.u, self.weights_x, self.weights_y)
        
        # Run Adam's optimizer
        self.LossArray = []
        for it in range(self.epochs_adam):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.idn_loss.outputs, tf_dict)
            self.callback(loss_value)
            if it % 10 == 0:
                print("It: %d, Loss = %.3e" %(it, loss_value))
        
        # Run L-BFGS optimizer
        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_loss.outputs],
                                      loss_callback = self.callback)
        
    def callback(self, loss):
        """"Callback for optimizers: save the current value of the loss function."""
        self.LossArray = self.LossArray + [loss]
    
    def save_loss(self):
        """Save the loss function in a file after training."""
        # Remove first L-BFGS loss
        self.LossArray.remove(self.LossArray[self.epochs_adam+1])
        its = [i for i in range(len(self.LossArray))]
        L = np.vstack((its, self.LossArray)).transpose()
        np.savetxt("%s/loss_%s.csv" % (self.path_training, self.activation_name), L, fmt = '%.4e', delimiter=',')
    
    def print_weights(self):
        """Print all the trainable weights."""
        print_weights(self)
            
    def save_results(self):
        """Save the Green's function evaluated at a grid in a csv file."""
        
        if self.dimension == 1:
            save_results(self)
            
        elif self.dimension == 2:
            for i in range(1,5):
                save_results(self, Green_slice=i)
        
        else:
            raise ValueError("Function not implemented for dimension greater than 2.")

    def plot_results(self):
        """Plot the learned Green's function and homogeneous solution."""
        plot_results(self)