import greenlearning as gl

#from evaluate_complex_hom import evaluate_complex

#from greenlearning.utils.backend import tf
#tf.set_random_seed(101)

def main():
    
    # Construct neural networks for G and homogeneous solution
    G_network = gl.matrix_networks([4] + [50] * 4 + [1], "rational", (1,1))
    U_hom_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1,))
    
    # Define the model
    model = gl.Model(G_network, U_hom_network)
    
    # Train the model on the selected dataset
    model.train("examples_2d/datasets/","poisson_disk")
    
    # Plot the results
    #model.plot_results()
    
    # Save the training loss
    #model.save_loss()
    
    #model.save_results()
    #evaluate_complex(model.sess, model.idn_N_pred, model.x, model.U_hom)
    
    # Close the session
    model.sess.close()

if __name__ == "__main__":
    
    main()