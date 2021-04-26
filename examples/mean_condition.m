function output_example = mean_condition()
% Mean one condition: http://www.chebfun.org/examples/ode-linear/NonstandardBCs.html

% Define the domain.
dom = [-1,1];

% Differential operator
N = chebop(@(x,u) diff(u,2)+x.^2.*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-1; mean(u)-1];

% Output
output_example = {N};
end