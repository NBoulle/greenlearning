from .utils.backend import tf
from .utils import config

class loss_function:
    """Loss function for learning Green's functions and homogeneous solutions.
    
    Inputs: matrices of neural networks G and N.
    """
    def __init__(self, G, N):
        self.G = G
        self.N = N
        self.build()

    @property
    def outputs(self):
        return self.loss
    
    def build(self):
        """Create Tensorflow placeholders and build the loss function."""
        
        # Spatial inputs
        self.xF = tf.placeholder(config.real(tf), [None, None])
        self.xU = tf.placeholder(config.real(tf), [None, None])
        
        # Spatial dimension
        d = self.N[0].layers[0]
        
        # Number of input data
        n_input = len(self.G[0])
        n_output = len(self.G)
        
        # Number of training points
        Nu = tf.shape(self.xU)[0]
        Nf = tf.shape(self.xF)[0]
        
        # Evaluation points for G
        Lu = []
        Lf = []
        for i in range(d):
            xG = tf.reshape(tf.repeat(tf.reshape(self.xU[:,i], (1, Nu)), Nf, 0), (Nu*Nf,1))
            yG = tf.reshape(tf.repeat(tf.reshape(self.xF[:,i], (Nf, 1)), Nu, 1), (Nu*Nf,1))
            Lu.append(xG)
            Lf.append(yG)
        training_G = tf.concat(Lu+Lf, 1)
        
        # Training data
        self.f = tf.placeholder(config.real(tf), shape=[None, None, n_input])
        self.u = tf.placeholder(config.real(tf), shape=[None, None, n_output])
        
        # Quadrature weights
        self.weights_x = tf.placeholder(config.real(tf), shape=[None, None])
        self.weights_y = tf.placeholder(config.real(tf), shape=[None, None, 1])
        
        # Multiply f by quadrature weights
        f_weights = tf.multiply(self.weights_y, self.f)
        
        # Compute the loss function
        # Loop over the number of outputs
        self.loss = 0
        for i in range(n_output):
            
            # Loop over the number of inputs
            self.loss_i = 0
            for j in range(n_input):
                # Evaluate Gij at all spatial points
                self.G_output = self.G[i][j].evaluate(training_G)
            
                # Compute integral of Gij*fj over y
                lossij = tf.reshape(self.G_output, (Nf, -1))
                
                # Transpose loss1, multiply by the vector F and divide by the number of samples
                self.loss_i = self.loss_i + tf.matmul(lossij, f_weights[:,:,j], transpose_a=True)
        
            # Get output of homogeneous solution
            self.N_output = self.N[i].evaluate(self.xU)
            
            # Difference with u
            loss_N = tf.repeat(self.N_output, tf.shape(self.u)[1], 1)
            relative_error = tf.divide(tf.reduce_sum(tf.multiply(self.weights_x, tf.square(self.u[:,:,i] - self.loss_i - loss_N)),0), \
                                       tf.reduce_sum(tf.multiply(self.weights_x, tf.square(self.u[:,:,i])),0))
            self.loss = self.loss + tf.reduce_mean(relative_error)
        
    def feed_dict(self, inputs_xU, inputs_xF, inputs_f, inputs_u, weights_x, weights_y):
        """Construct a feed_dict to feed values to TensorFlow placeholders."""
        
        feed_dict = {self.xU: inputs_xU, self.xF: inputs_xF, self.f: inputs_f, self.u: inputs_u, self.weights_x: weights_x, self.weights_y: weights_y}
        return feed_dict