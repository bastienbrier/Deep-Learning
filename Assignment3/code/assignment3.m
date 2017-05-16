clear all;clc;close all;

doSoftMax   = true;
doReLU      = true;
lambda      = .0001;
N_lambdas   = 5;
nb_folds    = 10;
nsamples    = 300;
do_print    = 1;

N_range = [5 10 20 50 100]; % different values of N to test
lambda_range = linspace(0.00001,0.0005,N_lambdas); % values of lambda

checkNNgradients(lambda,doSoftMax,doReLU)

cv_error = zeros(size(lambda_range,2),size(N_range,2)); % initialize matrix

for lbda=1:size(lambda_range,2)
    lambda = lambda_range(1,lbda); % regularization cost
    for n_neurons=1:size(N_range,2)
        N     = N_range(1,n_neurons); % number of neurons
        
        [features,labels,posterior] = construct_data(nsamples,'train','nonlinear');
        %% drop the constant term
        X = features([1,2],:)';
        %% labels change from 0,1 to 1,2 (required by the code below)
        y = labels' + 1;
        m = size(X, 1);

        %% Specify network architecture
        %% format: input dimension, # hidden notes at different layers, output dimension
        nnodes =  [2,N,2];

        %% initialize network parameters to random value
        randn('seed',0); % (but make it repeatable)

        nHidden    = length(nnodes)-1;
        initial_value = [];
        for l=1:nHidden
            %% add one for the constant component
            n_inputs        = nnodes(l) + 1;
            %% target neurons
            n_outputs       = nnodes(l+1);

            %% standard deviation of Gaussian distribution used for initialization
            sigma           = .1;
            WeightsLayer    = randn(n_inputs,n_outputs)*sigma;

            %% collate everything in one big parameter vector
            initial_value  = [initial_value;WeightsLayer(:)];
        end

        %% optimizer options
        options = optimset('MaxIter', 500);

        %% our optimization function (mincg) requres creating a pointer to
        %% the function that is being minimized
        error = zeros(1,nb_folds);
        for k = 1:nb_folds
            [training_set_inputs,training_set_targets,validation_set_features,...
                validation_set_targets] =  split_data(features,labels,nsamples,nb_folds,k);
            
            X_tr = training_set_inputs(1:2,:)'; % training set 
            y_tr = training_set_targets' + 1; % training labels
            X_CV = validation_set_features(1:2,:)'; % validation set
            y_CV = validation_set_targets' + 1; % validation labels
            
            costFunction = @(p) nnet(p, ...
                nnodes, X_tr, y_tr, lambda,...
                doSoftMax,doReLU);
            
            [nn_params, cost] = fmincg(costFunction, initial_value, options);
            if 1
                pred = nnet(nn_params, nnodes, X_CV, [], lambda, doSoftMax, doReLU);
                pred = (pred(:,1) < 0.5) + 1;
                fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == y_CV)) * 100);
                error(1,k) = 1 - mean(double(pred == y_CV));
            end
        end
        
        cv_error(lbda, n_neurons) = mean(error,2); % final score
        
        loc_x = [0:.01:1];
        loc_y = [0:.01:1];
        [grid_x,grid_y] = meshgrid(loc_x,loc_y);
        Xs      = [grid_x(:), grid_y(:)];
        [sv,sh] = size(grid_x);
        pred = nnet(nn_params,nnodes, Xs,[],lambda, doSoftMax, doReLU);
        pred = pred(:,1);
        figure,
        imshow(reshape(pred,[sv,sh]),[])
    end
end

% Plot the final matrix
figure,
imagesc(cv_error)
if do_print
    if doSoftMax
        if doReLU
            print('-djpeg', 'CV_softmax-relu')
        else
            print('-djpeg', 'CV_softmax')
        end
    else
        print('-djpeg', 'CV_relu')
    end
end
