%% Load the data
[Dtrain,Dtest]  = load_digit7;
[nsamples,ndimensions] = size(Dtrain);

% Save the data in a file or not
do_print = 1;

% Number of samples in test set
ntest = size(Dtest,1);

%% K-means for k=2
[centroid, distortion, affectation] = Kmeans(Dtrain, 2, 50, true);

%% Repeat 10 times
best_distortion_2 = 1e10;
best_centroid_2 = [];
best_affectation_2 = [];
for iterations=1:10
    [centroid, distortion, affectation] = Kmeans(Dtrain, 2, 50, false);
    str = ['Best distortion: ', num2str(best_distortion_2), ' - Current distortion: ', num2str(distortion)];
    disp(str); % display for progress
    if distortion < best_distortion_2
        best_distortion_2 = distortion; % affect the best distortion
        best_centroid_2 = centroid;
        best_affectation_2 = affectation;
    end
end

% calculate the error of the test
for i=1:size(Dtest,1)
    for j=1:2
        distance(i,j) = sqrt(sum((Dtest(i, :)-best_centroid_2(j,:)).^2));
    end
end
min_Vector = min(distance,[],2); % vector of the minimum distances
error_2 = sum(min_Vector); % assign the value of the error of test set

% Show the digit clusters
cluster_1 = reshape(best_centroid_2(1,:),[28,28]);
cluster_2 = reshape(best_centroid_2(2,:),[28,28]);
figure,
subplot(1,2,1); imshow(cluster_1);
title('Cluster 1');
subplot(1,2,2); imshow(cluster_2);
title('Cluster 2');
if do_print
        print ('K-means clusters', '-dpng'); % create file
end

%% Same procedure for k = 3; 4; 5; 10; 50; 100

distortion_vector_train = best_distortion_2; % keep track of train distortion
error_test = error_2; % keep track of test erro

for k=[3 4 5 10 50 100]
    best_distortion = 1e10;
    best_centroid = [];
    best_affectation = [];
    
    for iterations=1:10
        [centroid, distortion, affectation] = Kmeans(Dtrain, k, 50, false);
        str = ['k = ', num2str(k), ' - Best distortion: ', num2str(best_distortion), ' - Current distortion: ', num2str(distortion)];
        disp(str); % display for progress
        if distortion < best_distortion
            best_distortion = distortion; % assign the best distortion
            best_centroid = centroid;
            best_affectation = affectation;
        end
    end
    
    % add best distortion to the vector of distortion in train set
    distortion_vector_train=[distortion_vector_train, best_distortion];
    
    % calculate the error of the test
    for i=1:size(Dtest,1)
        for j=1:k
            distance(i,j) = sqrt(sum((Dtest(i, :)-best_centroid(j,:)).^2));
        end
    end
    min_Vector = min(distance,[],2); % vector of the minimum distances
    error = sum(min_Vector); % assign the value of the error of test set
    error_test = [error_test, error]; % add to the vector
end

%% Plots
figure,
plot([2 3 4 5 10 50 100], error_test, 'color', [0,0,1], 'linewidth', 2);
hold on,
plot([2 3 4 5 10 50 100], distortion_vector_train, 'color', [1,0,0], 'linewidth', 2)
xlabel('Number of clusters','fontsize',16);
ylabel('Total error','fontsize',16);
legend('Test error', 'Train error');
if do_print
        print ('K-means error', '-dpng'); % create file
end