function output_example = interior_layer()
% Interior layer equation

% Define the domain.
dom = [0,1];

% Parameter of the equation
D = 5e-3;

% Differential operator
N = chebop(@(x,u)D*diff(u,2)+x*(x^2-0.5)*diff(u)+3*(x^2-0.5)*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N};
end