function output_example = nonlinear_SL()
% Nonlinear Sturm-Liouville equation

% Define the domain.
dom = [0,2*pi];

% Parameter of the equation
p = chebfun(@(x)0.5*sin(x)-3,dom); 
q = chebfun(@(x)0.6*sin(x)-2,dom);
eps = 0.4;

% Differential operator
N = chebop(@(x,u) diff(-p*diff(u))+q*(u+eps*u^3), dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N, "nonlinear"};
end