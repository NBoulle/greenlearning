function output_example = ODE_system()
% System of ODEs

% Define the domain.
dom = [-1, 1];

% Differential operator
N = chebop(@(x,u,v) [diff(u,2)-v; -diff(v,2)+x.*u], dom);

% Boundary conditions
N.bc = @(x,u, v) [u(-1)-1; u(1)+1; v(-1)+2; v(1)+2];

% Output
output_example = {N};
end