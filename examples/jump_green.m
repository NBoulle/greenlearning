function output_example = jump_green()
% 2nd order ODE with a jump condition
% http://www.chebfun.org/examples/ode-linear/JumpGreen.html

% Define the domain.
dom = [0,1];

% Parameter of the equation
eta = 0.2;

% Differential operator
N = chebop(@(x,u) eta*diff(u,2) + diff(u), dom);

% Boundary conditions
N.lbc = 0; N.rbc = 0;
N.bc = @(x,u) [feval(u,.7,'left')-2 ; feval(u,.7,'right')-1];

% Output
output_example = {N};
end