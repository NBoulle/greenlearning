function output_example = potential_barrier()
% Potential barrier: http://www.chebfun.org/examples/ode-linear/LeeGreengardODEs.html

% Define the domain.
dom = [-1,1];

% Parameter of the equation
ep = 5e-3;

% Differential operator
N = chebop(@(x,u) ep*diff(u,2)+(x.^2-0.25).*u,dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-1; u(dom(2))-2];

% Output
output_example = {N};
end