function [J, grad] = nnet(nn_params,nnodes, X, y, lambda,doSoftMax,doReLU)
%nnet Implements a neural network.
%   [J, grad] = nnet(nn_params,nnodes, X, y, lambda,doSoftMax,doReLU)
%
% The function implements two functionalities
% Test mode:
%    if y is an empty matrix, J provides the outputs of the network
% Train mode:
%    if y contains ground-truth labels, J provides the objective of the network
%    if a second output argument is requested, grad provides the gradient
%    w.r.t the network parameters
%
% nnodes indicates the number of nodes in the network in terms of a vector
% e.g. nnodes = [2,10,10,2] indicates we have 2-D inputs, two layers of
% hidden neurons with 10 neurons each, and 2-D outputs.
%
% nn_params contains all of the network parameters, folded into a single
% vector; for the case above, we would have a 2x10 + 10x10 + 10x2 = 140-D
% vector
% If grad (2nd output) is requested, the dimensionality should be the same.
%
% lambda is the weight of the regularizer (common across all layers)
% doSoftMax,doReLU are booleans (false/true) indicating whether we
% employ the SoftMax unit and/or ReLUs respectively.
%

input_layer_size = nnodes(1);
nHidden = length(nnodes)-1;
num_labels = nnodes(end);

%% unfold parameters into layer-specific weights
offset = 0;
for l=1:nHidden
    n_inputs    = nnodes(l) + 1; %% add one for constant component
    n_outputs   = nnodes(l+1);
    Weights{l}  = reshape(nn_params(offset + [1:n_outputs * n_inputs]),n_outputs,n_inputs);
    offset      = offset + n_inputs*n_outputs;
    
    %% allocate space for gradients
    GradWeights{l} = zeros(size(Weights{l}));
end

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly
J = 0;

%% Network in Forward mode

%% neuron outputs from previous layer
%inputs{1} = [ones(m,1) X];
inputs{1} = [ones(m,1) X];
for l=1:nHidden
    z = [inputs{l}] * Weights{l}';
    activations{l} = z;
    if l<nHidden        
        if doReLU
            o = ReLU(z);
        else
            o = sigmoid(z);
        end
        inputs{l+1} = [ones(m,1) o];        
    else
        %% for the very last layer:
        %% compute class posteriors
        if doSoftMax
            Posteriors = softmax(z);
        else
            Posteriors = sigmoid(z);
        end
    end
end

if isempty(y)
    %% if no ground-truth is provided, return posteriors
    J = Posteriors; return;
end

%% if ground truth is provided, go on

%% First, compute objective

for k = 1:num_labels
    yk = y == k;
    Posteriorsk = Posteriors(:, k);
    if doSoftMax
        Jk = 1 / m * sum(-yk .* log(Posteriorsk));
    else
        Jk = 1 / m * sum(-yk .* log(Posteriorsk) - (1 - yk) .* log(1 - Posteriorsk));
    end
    J = J + Jk;
end

regularization = 0;
for l=1:nHidden
    regularization = regularization + sum(sum(Weights{l}(:,2:end).^2));
end
J = J + lambda*regularization;

if nargout==2
    %% Then, compute gradient
    for t = 1:m
        %% gradient signal coming from top layer into network
        
        for k = 1:num_labels
            yk = y(t) == k;
            delta_top(k,1) = Posteriors(t, k) - yk;
        end
        
        %% back-propagate it
        for l=(nHidden):-1:1
            %% symbols for any neuron:
            %% activation -  nonlinearity  - output
            if (l<nHidden)
                %% partial derivative w.r.t. node activations
                if doReLU
                    d_output_d_activation = reluGradient([activations{l}(t, :)])';
                else
                    d_output_d_activation = sigmoidGradient([activations{l}(t, :)])';
                end
                d_loss_d_activation = d_loss_d_output .* d_output_d_activation;
            else
                d_loss_d_activation = delta_top;
            end
            GradWeights{l}         = GradWeights{l} + d_loss_d_activation * [inputs{l}(t, :)];
            d_loss_d_output_below  = Weights{l}'  *  d_loss_d_activation;
            d_loss_d_output        = d_loss_d_output_below(2:end);
        end
    end
    
    
    for l=1:length(GradWeights)
        GradWeights{l} = GradWeights{l} / m;
        GradWeights{l}(:, 2:end) = GradWeights{l}(:, 2:end) + 2*lambda * Weights{l}(:, 2:end);
    end
    
    %% fold gradients into single vector
    grad = [];
    for l=1:length(GradWeights)
        grad = [grad ; GradWeights{l}(:)];
    end
end
