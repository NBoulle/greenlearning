from .backend import tf

def print_weights(model):
    """Print all the trainable weights."""
    
    # Get all the variables
    variables_names = [v.name for v in tf.trainable_variables()]
    values = model.sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)