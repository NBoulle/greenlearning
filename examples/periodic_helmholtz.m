function output_example = periodic_helmholtz()
% Helmholtz equation

% Define the domain.
dom = [0,1];

% Parameter of the equation
K = 15;

% Differential operator
N = chebop(@(x,u) diff(u,2)+K^2*u, dom);

% Boundary conditions
N.bc = 'periodic';

% Kernel frequency for f
lambda = 0.2;

% Output
output_example = {N, "lambda", lambda};
end