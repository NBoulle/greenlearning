function output_example = laplace()
% Laplace equation

% Define the domain.
dom = [0,1];

% Differential operator
N = chebop(@(x,u) -diff(u,2), dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Exact Green's function
G = 'x*(1-y)*(x<=y) + y*(1-x)*(x>y)';

% Output
output_example = {N, "ExactGreen", G};
end