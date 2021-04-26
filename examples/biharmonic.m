function output_example = biharmonic()
% Biharmonic equation

% Define the domain.
dom = [0,3];

% Differential operator
N = chebop(@(x,u)diff(u,4), dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(1); u(2); u(dom(2))];

% Kernel frequency for f
lambda = 0.25;

% Output
output_example = {N, "lambda", lambda};
end