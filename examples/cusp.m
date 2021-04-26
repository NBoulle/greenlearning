function output_example = cusp()
% Cusp: http://www.chebfun.org/examples/ode-linear/LeeGreengardODEs.html

% Define the domain.
dom = [-1,1];

% Parameter of the equation
ep = 1e-6;

% Differential operator
N = chebop(@(x,u) ep*diff(u,2)+x.*diff(u)-0.5*u,[-1 0 1]);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-1; u(dom(2))-2];

% Kernel frequency for f
lambda = 0.1;

% Output
output_example = {N, "lambda", lambda};
end