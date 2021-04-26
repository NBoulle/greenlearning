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

% Exact Green's function
G = sprintf('(%d*np.sin(%d))**(-1)*np.sin(%d*x)*np.sin(%d*(y-1))*(x<=y)+(%d*np.sin(%d))**(-1)*np.sin(%d*y)*np.sin(%d*(x-1))*(x>y)',K,K,K,K,K,K,K,K);

% Output
output_example = {N, "ExactGreen", G};
end
