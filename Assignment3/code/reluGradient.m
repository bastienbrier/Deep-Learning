function g = reluGradient(z)
%RELUGRADIENT returns the gradient of the ReLU function
%evaluated at z

g = (z > 0);
end