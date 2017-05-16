%% Visualize or not
do_see = 1;

% Create file or not
do_print = 0;

%% Load the data
[Dtrain,Dtest]  = load_digit7;

whos
[nsamples,ndimensions] = size(Dtrain);

% Number of samples in test set
ntest = size(Dtest,1);

%% Mean image
meanDigit = mean(Dtrain,1);
meanImage = reshape(meanDigit,[28,28]);
figure,imshow(meanImage);

% Covariance matrix
covDigits = cov(Dtrain);

% Substract the mean from the data
Xtrain = Dtrain - repmat(meanDigit, nsamples,1);
Xtest = Dtest - repmat(meanDigit, ntest, 1);

%% Find the eigenvalues and eigenvectors and calculate the error
error_vector = zeros(1,10);
error_vector_train = zeros(1,10);

for dimensionality = 1:10
    [eigvec,eigvals] = eigs(covDigits,dimensionality);
    
    % calculate test error
    exp_coeff = Xtest * eigvec; % coefficients of the image projected on the eigenvector
    reconstructed_test = exp_coeff * eigvec' + meanDigit; % reconstructed image after PCA
    euc_norm = sqrt(sum((Dtest-reconstructed_test).^2,2)); % euclidean norm of each image (each row)
    error = sum(euc_norm); % sum of the norms
    error_vector(1, dimensionality) = error; % add it to the error vector
    
    % same for train error
    exp_coeff_train = Xtrain * eigvec; % coefficients of the image projected on the eigenvector
    reconstructed_train = exp_coeff_train * eigvec' + meanDigit; % new image after PCA
    euc_norm_train = sqrt(sum((Dtrain-reconstructed_train).^2,2)); % euclidean norm of each image (each row)
    terror = sum(euc_norm_train); % sum of the norms
    error_vector_train(1, dimensionality) = terror; % add it to the error vector
    
    % Visualization
    if do_see
        number = 1010; % pick an image
        orig_example = reshape(Xtest(number,:), [28,28]); % test image
        new_example = reshape(reconstructed_test(number,:),[28,28]); % new test image after PCA
        figure,
        subplot(1,2,1); imshow(orig_example);
        subplot(1,2,2); imshow(new_example);
    end
end


%% Showing the final eigenvectors:
figure,
for i = 1:dimensionality
    subplot(4,3,i); imshow(reshape(eigvec(:,i),[28,28]),[]);
end

%% Plot
figure,
plot(error_vector,'color', [0,0,1],'linewidth', 2);
hold on,
plot(error_vector_train,'color', [1,0,0], 'linewidth', 2)
xlabel('Number of eigenvalues','fontsize',16);
ylabel('Total error','fontsize',16);
legend('Test error', 'Train error');
if do_print    
    print ('PCA error', '-dpng'); % create file
end
