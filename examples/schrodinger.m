function output_example = schrodinger()
% Schrodinger equation

% Define the domain.
dom = [-3,3];

% Parameter of the equation
h = 0.1;
V = chebfun(@(x) x^2+1.5*exp(-(x/0.25)^4), dom);

% Differential operator
N = chebop(@(x,u)-h^2*diff(u,2) + V(x)*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N};
end