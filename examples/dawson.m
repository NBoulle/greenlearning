function output_example = dawson()
% ODE with interior point condition
% http://www.chebfun.org/examples/ode-linear/DawsonIntegral.html

% Define the domain.
dom = [-2,2];

% Differential operator
N = chebop(@(x,u) diff(u,1) + 2*x.*u, dom);

% Boundary conditions
N.bc = @(x,u) u(0);

% Output
output_example = {N};
end