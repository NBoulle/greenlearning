function output_example = identity()
% Identity operator

% Define the domain.
dom = [-1,1];

% Differential operator
N = chebop(@(x,u) u, dom);

% Output
output_example = {N};
end