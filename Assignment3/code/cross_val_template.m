%% Instructions:
% This code is a template, it is just like the code for cross validation 
% you used in previous assignments, it should be very familiar to you by now!
% You should take this code and incorporate it into assignment5.m, adapting 
% it as needed.

N_Ns = 20;
N_lambdas = 20;
N_range = linspace( , ,N_Ns); % Enter suitable range (N must be an integer)
lambda_range  = logsample( , ,N_lambdas); % Enter suitable range

cv_error = zeros(size(lambda_range,2),size(N_range,2));

for i=1:size(lambda_range,2)
    lambda = lambda_range(1,i);
    for j=1:size(N_range,2)
        N     = N_range(1,j);
        K     = 10;
        error = zeros(1,K);
        
        for k=1:K
            %% TEMPLATE FOR CROSS-VALIDATION CODE 
            %split data into training set and validation set
            [training_set_inputs,training_set_targets,validation_set_features,...
                validation_set_targets] =  split_data(features,labels,nsamples,K,k);

            %Training the neural network
            
                
            %Estimate the error for each validation_run
            
            
            
            
        end
        %The generalization error is the mean of the error
        cv_error(i,j)=mean(error,2);
    end
end
colormap hsv;
figure,imagesc(cv_error)


%Pick the best lambda and N - those that minimize the cv_error





