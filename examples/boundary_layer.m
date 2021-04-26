function output_example = boundary_layer()
% Boundary layer : http://www.chebfun.org/examples/ode-linear/BoundaryLayer.html

% Define the domain.
dom = [0,1];

% Parameter of the equation
eps = 0.01;

% Differential operator
N = chebop(@(u) -eps*diff(u,2) - diff(u),dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N};
end