function [x_L, P_h]    = block_gibbs(x_0, W, L);
% Inputs : x_0 : inputs
%          W   : weights
%          L   : length of Block-Gibbs sampling
% Principle : we start by conditioning on x0 and sampling h0 from its conditional distribution, then
% conditioning on h0 and sampling x1 and then repeating L times.
x = x_0;

for l = 1:L
    % Sample h from x
    p_h = 1 ./ (1+exp(-2*x*W));
    threshold_h = double(p_h > rand(size(p_h)));
    % = P(h=1|x^m;W) 
    h = 2 * threshold_h - 1;
    
    % Sample x from h
    p_x = 1 ./ (1+exp(2*h*W'));
    threshold_x = double(p_x > rand(size(p_x)));
    x = 2 * threshold_x - 1;
end

% Final value of x
x_L = x;
% Final value of P_h
P_h = 1 ./ (1+exp(-2*x_L*W));