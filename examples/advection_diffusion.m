function output_example = advection_diffusion()
% Advection-diffusion equation

% Define the domain.
dom = [0,1];

% Parameter of the equation
D = 1/4;

% Differential operator
N = chebop(@(x,u) D*diff(u,2)+diff(u)+u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1))-1; u(dom(2))+2];

% Exact Green's function
if D == 1/4
    G = '4*np.exp(-2*(x-y))*((y-1)*x*(x<=y)+(x-1)*y*(x>y))';
else
    G = '0';
end

% Output
output_example = {N, "ExactGreen", G};
end