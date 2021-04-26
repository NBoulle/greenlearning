function output_example = cubic_helmholtz()
% Cubic Helmholtz equation

% Define the domain.
dom = [0,2*pi];

% Parameter of the equation
alpha = -1; 
eps = -0.3;

% Differential operator
N = chebop(@(x,u) diff(u,2)+alpha*u+eps*u^3, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N, "nonlinear"};
end