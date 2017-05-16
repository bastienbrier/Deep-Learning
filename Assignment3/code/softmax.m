function g = softmax(z)
%SOFTMAX Compute softmax functoon
%   J = SOFTMAX(z) computes the softmax of z.

g = exp(z) ./ sum(exp(z),2);
end

