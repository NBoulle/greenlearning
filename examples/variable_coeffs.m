function output_example = variable_coeffs()
% Variable coefficients

% Define the domain.
dom = [-1,1];

% Differential operator
N = chebop(@(x,u) 0.1*diff(u,2) + sin(x^2)*diff(u) + 10*(x>=0)*cos(10*pi*x)*u,dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-1; u(dom(2))+2];

% Output
output_example = {N};
end