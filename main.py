import greenlearning as gl

def main():
    
    # Construct neural networks for G and homogeneous solution
    G_network = gl.matrix_networks([2] + [50] * 4 + [1], "rational", (1,1))
    U_hom_network = gl.matrix_networks([1] + [50] * 4 + [1], "rational", (1,))
    
    # Define the model
    model = gl.Model(G_network, U_hom_network)
    
    # Train the model on the selected dataset
    model.train("examples/datasets/","helmholtz")
    
    # Plot the results
    model.plot_results()
    
    # Save the results
    model.save_results()
    
    # Close the session
    model.sess.close()

if __name__ == "__main__":
    
    main()
