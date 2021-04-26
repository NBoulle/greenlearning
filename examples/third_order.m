function output_example = third_order()
% Third order equation

% Define the domain.
dom = [0,2];

% Differential operator
N = chebop(@(x,u)diff(u,3)+u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(1); u(dom(2))];

% Output
output_example = {N};
end