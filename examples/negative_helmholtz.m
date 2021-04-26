function output_example = negative_helmholtz()
% Helmholtz equation

% Define the domain.
dom = [0,1];

% Parameter of the equation
K = 8;

% Differential operator
N = chebop(@(x,u) diff(u,2)-K^2*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Exact Green's function
G = sprintf('(%d*np.sinh(%d))**(-1)*np.sinh(%d*x)*np.sinh(%d*(y-1))*(x<=y) + (%d*np.sinh(%d))**(-1)*np.sinh(%d*y)*np.sinh(%d*(x-1))*(x>y)',K,K,K,K,K,K,K,K);

% Output
output_example = {N, "ExactGreen", G};
end