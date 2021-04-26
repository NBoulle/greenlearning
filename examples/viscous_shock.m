function output_example = viscous_shock()
% Viscous shock equation
% http://www.chebfun.org/examples/ode-linear/LeeGreengardODEs.html

% Define the domain.
dom = [-1,1];

% Parameter of the equation
ep = 1e-3;

% Differential operator
N = chebop(@(x,u) ep*diff(u,2) + 2*x.*diff(u),dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))+1;u(dom(2))-1];

% Kernel frequency for f
lambda = 0.1;

% Output
output_example = {N, "lambda", lambda};
end