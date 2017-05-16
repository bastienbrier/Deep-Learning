function [centroid, distortion, cluster] = Kmeans(data, k, maxIteration, do_disp)

    % data: data to analyze
    % k: number of clusters
    % maxIteration: maximum number of iteration
    % do_disp: display the intermediate distortion or not

do_print = 0; % create the file or not

%% Initialize centroid with random guess
w_index = randi(size(data,1), k, 1);    % Pick ai_K values
centroid = data(w_index, :);
[centers,junk]=datasample(1:k,size(data,1));
centers = centers';

%% Update centroid

% Make a local copy of the data
w_Data = data;
distortion_vector = []; % initialize an empty vector

for w_iter = 1:maxIteration
    
    %% Evaluate distance to each centroid
    distance = zeros(size(w_Data,1), k);
    % loop over the rows of w_Data and the three centroids to compute the
    % distance regarding the 3, and save it in the w_Distance matrix
    for i=1:size(w_Data,1)
        for j=1:k
            distance(i,j) = sqrt(sum((w_Data(i, :)-centroid(j,:)).^2));
        end
    end
    %% Find the minimum value 
    % find the index of the minimum of the three values of each row of
    % w_Distance (need to take transpose for min function to work)
    [minimum_dist, best_centroid] = min((distance)');
    
    centers(:,1) = best_centroid'; % affect new labels to the data
    cluster = centers;
    
    %% Calculate the total distortion
    min_Vector = min(distance,[],2); % vector of the min distances
    distortion = sum(min_Vector);
    distortion_vector = [distortion_vector, distortion]; % add the distortion to the vector of distortions
    str = ['Distortion: ', num2str(distortion)]; % array to display value
    if do_disp
        disp(str); % display the value if ai_print is True
    end
    
    %% Update Centroid Value
    for c=1:k
        count = 0;
        for i=1:size(w_Data,1)
            % add the coordinates of the points if they are labeled in
            % centroid c
            if centers(i,1) == c
                centroid(c,:) = centroid(c,:) + w_Data(i,:);
                count = count + 1;
            end
        end
        % and then divide by the number of points labeled
        if count > 0
            centroid(c,:) = centroid(c,:) ./ count;
        end
    end
    
    %% Break if convergence is reached
    if (w_iter > 2)    
        one_difference = distortion_vector(1,w_iter) - distortion_vector(1,w_iter-1);
        two_difference = distortion_vector(1,w_iter) - distortion_vector(1,w_iter-2);
        if (abs(one_difference) < 0.0001) & (abs(two_difference) < 0.0001)
            break;
        end
    end
end

%% Plot the distortion if do_disp
if do_disp
    figure,
    plot([1:10], distortion_vector(1,1:10), 'Color',[0,0,1],'linewidth', 2);
    xlabel('Number of iteration','fontsize',16);
    ylabel('Distortion','fontsize',16);
    
    if do_print
        print ('K-means iteration', '-dpng'); % create file
    end
end

end
