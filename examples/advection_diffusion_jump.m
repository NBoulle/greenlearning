function output_example = advection_diffusion_jump()
% Advection-diffusion for x>=0 : http://www.chebfun.org/examples/ode-linear/AdvDiffJump.html

% Define the domain.
dom = [-1,1];

% Differential operator
N = chebop(@(x,u) 0.1*diff(u,2) + (x>=0).*diff(u),dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-2; u(dom(2))+1];

% Output
output_example = {N};
end