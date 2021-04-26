function output_example = airy_equation()
% Advection-diffusion equation

% Define the domain.
dom = [0,1];

% Parameter of the equation
K = 10;

% Differential operator
N = chebop(@(x,u) diff(u,2)-K^2*x*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N};
end