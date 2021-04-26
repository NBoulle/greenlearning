Creating a training dataset
===========================

In this section, we explain how to generate a training dataset in MATLAB to learn the Green's function associated to an ODE or a system of ODEs.

This step requires the MATLAB package called Chebfun (see `<https://www.chebfun.org/download/>`_ for installation instructions).


Definition of the differential operator
---------------------------------------

The definition of the differential operator can be done by creating a MATLAB script in a folder ``examples/``.
In the following example, we add a script called ``helmholtz.m`` in the folder with the following content:

.. code-block:: matlab

    function output_example = helmholtz()
    % Helmholtz equation
    
    % Define the domain.
    dom = [0,1];
    
    % Parameter of the equation
    K = 15;
    
    % Differential operator
    N = chebop(@(x,u) diff(u,2)+K^2*u, dom);
    
    % Boundary conditions
    N.bc = @(x,u) [u(dom(1)); u(dom(2))];
        
    % Output
    output_example = {N};
    end

Here, we define a Helmholtz operator :math:`\mathcal{L}u = \frac{d^2u}{dx^2}+K^2 u` with Helmholtz frequency :math:`K=15`, on the interval :math:`[0,1]`, with homogeneous Dirichlet boundary conditions.

- One could, for instance, define the boundary conditions to be :math:`u(0)=-1`, :math:`u(1)=2` as follows:

    .. code-block:: matlab
    
     % Boundary conditions
     N.bc = @(x,u) [u(dom(1))+1; u(dom(2))-2];

- It is also possible to impose :math:`u(0)=0`, :math:`u'(0)=1` with the following line:

    .. code-block:: matlab
    
     % Boundary conditions
     N.bc = @(x,u) [u(dom(1)); feval(diff(u),dom(1))-1];

If the exact Green's function is known, one can in addition provide its expression (as a ``string`` in ``numpy`` format) to compare with the learned Green's function:

.. code-block:: matlab

    % Exact Green's function
    G = sprintf('(%d*np.sin(%d))**(-1)*np.sin(%d*x)*np.sin(%d*(y-1))*(x<=y)+(%d*np.sin(%d))**(-1)*np.sin(%d*y)*np.sin(%d*(x-1))*(x>y)',K,K,K,K,K,K,K,K);
    
    % Output
    output_example = {N, "ExactGreen", G};

System of differential operators
--------------------------------

The syntax is similar to define a system of differential operators. In this example, we define the following system:

.. math::

    \mathcal{L}(u,v) = \left(\begin{array}{c} \frac{d^2u}{dx^2}-v\\
                                     -\frac{d^2v}{dx^2}+xu
                       \end{array}\right),

on the domain :math:`[-1,1]` with boundary conditions:

.. math::

    u(-1)=1,\,u(1)=-1,\,v(-1)=v(1)=-2.

.. code-block:: matlab

    function output_example = ODE_system()
    % System of ODEs
    
    % Define the domain.
    dom = [-1, 1];
    
    % Differential operator
    N = chebop(@(x,u,v) [diff(u,2)-v; -diff(v,2)+x.*u], dom);
    
    % Boundary conditions
    N.bc = @(x,u, v) [u(-1)-1; u(1)+1; v(-1)+2; v(1)+2];
    
    % Output
    output_example = {N};
    end

Generating the dataset
----------------------

Make sure that the MATLAB codes `generate_gl_example.m <https://github.com/NBoulle/greenlearning/blob/main/generate_gl_example.m>`_ and `generate_gl_datasets.m <https://github.com/NBoulle/greenlearning/blob/main/generate_gl_datasets.m>`_ are present in the parent directory of ``examples/`` and run the following command in a MATLAB terminal:

.. code-block:: matlab

    generate_gl_example("helmholtz");

Alternatively, all the datasets corresponding to the examples in the folder ``examples/`` can be generated as follows:

.. code-block:: matlab

    generate_gl_datasets();

The datasets are saved as ``.mat`` files at the location ``examples/datasets/``.

- By default, the code generates :math:`100` sampled functions :math:`f` from a squared-exponential Gaussian process, and solve the equation :math:`\mathcal{L}u=f` for the different forcing terms to obtain the training solutions :math:`u`. Then, the forcing terms are evaluated at :math:`200` uniform points in the domain :math:`\Omega`, while the solutions are evaluated at :math:`100` points. 

- It is possible to edit the MATLAB script `generate_gl_example.m <https://github.com/NBoulle/greenlearning/blob/main/generate_gl_example.m>`_ to add noise to the output functions, or change the different parameters such as number of forcing terms, spatial points, ...