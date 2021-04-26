function output_example = nonlinear_biharmonic()
% Nonlinear biharmonic

% Define the domain.
dom = [0,2*pi];

% Parameter of the equation
p = -4;
q = 2;
eps = 0.4;

% Differential operator
N = chebop(@(x,u) diff(-p*diff(u,2),2)+q*(u+eps*u^3), dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2));feval(diff(u),dom(1)); feval(diff(u),dom(2))];

% Output
output_example = {N, "nonlinear"};
end