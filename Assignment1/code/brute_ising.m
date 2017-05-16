clear all

%% Print images or not 
do_print = 0; % set to 1 to print  jpegs  (do_print=0 makes things faster)

%% first or second part of 3rd exercise
part_1 = 0;

%% vertical & horizontal size of lattice
sv = 4;
sh = 4;

%% connection strength (>0 favors similar neighboring spins)
J           = .5;
%% matrix containing all possible connections
Js          = zeros(sv*sh,sv*sh);
%% connect only grid-based neighbors
for h = [1:sh]
    for v= [1:sv]
        neighs = [];
        pt = (h-1)*sv + v;
        %%
        if v>1
            neighs = [neighs, (h-1)*sv + v-1];
        end
        if v<sv
            neighs = [neighs, (h-1)*sv + v+1];
        end
        if h>1
            neighs = [neighs, (h-2)*sv + v];
        end
        if h<sh
            neighs = [neighs, (h)*sv + v];
        end
        Js(pt,neighs) = J;
        Js(neighs,pt) = J;
    end
end

%% exhaustive enumeration of possible states
nnodes      = sv*sh;
states      = get_all_states(nnodes);

%%----------------------------------------------------
%% Botlzmann-Gibbs distribution
%%----------------------------------------------------

    %% construct [1 x 2^{nnodes}] vector ener such that
    %% ener(1,i) measures the energy of states(:,i)
    ener = zeros(size(states,1),1);
    for i=1:size(states,1)
        ener(i,1) = (-1/2)*states(i,:)*Js*states(i,:)';
    end
    %% compute partition function
    Z           = sum(exp(-ener));
    %% compute probability of state i in [1 x 2^{nnodes}] vector P
    P           = (1/Z)*exp(-ener);

if do_print
    figure(1)
    plot(P);
    xlabel('state','fontsize',16); ylabel('P(state)','fontsize',16);
    print('-djpeg','pstate');
    
    [~,i_max]   = max(ener);
    [~,i_min]   = min(ener);
    
    figure(2)
    subplot(1,2,1);
    imshow(reshape(states(i_max,:),[sv,sh]),[-1,1])
    title('max-energy state')
    subplot(1,2,2);
    imshow(reshape(states(i_min,:),[sv,sh]),[-1,1])
    title('min-energy state')
    print('-djpeg','energystate');
end

%% identify all (i,j) pairs showing up in the energy function
[is,js]     = find(Js);
nedges      = length(is);

if part_1
    %%----------------------------------------------------
    %% Brute-force computation of feature expectations
    %% E_{ij} = <x_i x_j>_{P(x)} = sum_{x} P(x) [x_i x_j]
    %%----------------------------------------------------
    for k=1:nedges
            E_ij(k) = sum(states(:,is(k)).*states(:,js(k)).*P);
    end
    if do_print
        %save('E_ij','E_ij')
        figure(3);
        plot(E_ij,'linewidth',2);
        title('E_{ij} = <x_i x_j>_{P(x)}','fontsize',16);
        
        print('-djpeg','Expectations');
    end
else
    %% expectations obtained for some unknown value of J
    t=load('E_ij');
    E_ij = t.E_ij;
end

%% ------------------------------------------------------------
%% Monte Carlo estimation of feature expectations with Gibbs sampling
%% E_{ij} = sum_{k} x^k_i x^k_j,   x^k ~ P(x)
%% ------------------------------------------------------------


if part_1
    %% single run of Gibbs sampling, just to get expectations
    it_train  = 0;
else
    %% repeated steps [expectation - parameter update]
    it_train  = 10;
    %% inital value of parameter (to be updated by gradient ascent)
    J_        = .5; 
end

for it_out = [1:(it_train+1)];
    %% set it to 1 to display some of the sampled states
    do_viz = 1;
    
    %% total number of Gibbs sampling iterations
    nits_tot    = 210000;
    %% 'sacrificed' iterations for burn-in
    burn_in     = 1000;
    %% consecutive samples are not independent -use every 100th sample
    %% for estimation
    step_sample = 100;
    samples_tot = (nits_tot-burn_in)/step_sample;
    
    %% running estimate of the expectation
    avg_ij      = zeros(samples_tot,nedges);
    %% current sum of state products
    sum_ij      = zeros(nedges,1);
    
    %% number of terms that have been added so far
    it_sum = 0;
    
    %% set to 1 if you want to see your Gibbs samples
    doviz = 0;
    if part_1
        %% displays some of the sampled states
        fprintf(2,'Gathering samples .. \n');
    else
        Js(Js~=0) = J_;
    end
    
    %% to ensure we get reproducible results
    rand('seed',0);
    
    %% randomly initialize network's state
    X           = 2*double(rand(nnodes,1)>.5)-1;
    
    for it = 1:nits_tot
        %% select a pixel at random
        ix = ceil( sh * rand(1) );
        iy = ceil( sv * rand(1) );
        %% find its index in 2D array
        position    = iy + sv*(ix-1);
        
        %% edge weights connecting 'position' to all other nodes
        neighsCon   = Js(:,position);
        
        %% ------------------------------------------------------------
        %% Gibbs sampling, inner loop
        %% ------------------------------------------------------------
            %% energy if we assign state +1 to 'position'
            ener_p1 =    -neighsCon'*X;
            %% ditto for state -1
            ener_m1 =    -ener_p1;
            
            %% posterior probability that 'position' will be +1
            p1          = exp(-ener_p1)/(exp(-ener_p1)+exp(-ener_m1));
            
            %% decide whether to set 'position' to +1 or -1
            state       =  2*double(rand(1)<p1)-1;


        X(position) = state;
        if it>burn_in
            if mod(it,step_sample)==0
                it_sum              = it_sum + 1;
                for k=1:nedges
                    sum_ij(k)           = sum_ij(k) + X(is(k))*X(js(k));
                    avg_ij(it_sum,k)    = sum_ij(k)/it_sum;
                end
                if part_1
                    fprintf(2,'.');
                    if mod(it_sum,40)==0
                        fprintf(2,'\n');
                    end
                end
                
                if doviz
                    warning off;
                    figure(4); clf
                    imshow(reshape(X,[sv,sh]),[-1,1]);
                    pause;
                end
            end
        end
    end
    
    % Sampling and gradient descent
    if part_1
        fprintf(2,'Done sampling \n');
    else
        fprintf(2,'Done sampling -updating parameters\n');
        avg_final       = avg_ij(it_sum,:);
        Stats(it_out,:) = avg_final;
            % after having tested many steps, 0.015 is the one with the most precise solution 
            % (lower steps don't come to a solution in 11 iterations)
            J_ = J_ - 0.015*(sum(avg_final - E_ij)); %% gradient ascent with 0.015 step

        J_track(it_out) = J_;
        fprintf(2,'.\n');
    end
end


% Plots Part-2
if part_1
    %% show intermediate expectations
    shown    = ceil(linspace(1,it_sum,11));
    shown(1) = []; nshown = length(shown);
    colors   = hsv(nshown);
    
    % Estimates MCMC
    figure(5); clf
    for k=1:nshown
        str{k}  = sprintf('Iteration: %i\n',shown(k));
        plot(avg_ij(shown(k),:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    str{nshown+1} = 'Brute Force';
    plot(E_ij,'k','linewidth',2);
    axis([0,length(E_ij),0,1])
    legend(str,'location','southwest');
    if do_print
        print('-djpeg','Estimates')
    end
    
    % Comparison Monte Carlo and Brute Force
    figure(6); clf;
    for k=nshown
        plot(avg_ij(shown(k),:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    plot(E_ij,'k','linewidth',2);
    axis([0,length(E_ij),0,1])
    legend({'Monte Carlo','Brute Force'})
    if do_print
        print('-djpeg','MCvsBF')
    end

% Plots Part-3
else
    nshown = 10;
    colors = hsv(nshown + 2);

    % Updates of the estimations per iteration of the gradient
    figure(7); clf;
    for k=[1:11]
        str{k}  = sprintf('Iteration %d',k);
        plot(Stats(k,:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    str{k+1} = 'Final';
    plot(E_ij,'k','linewidth',2);
    legend(str,'location','southwest');
    axis([0,length(E_ij),0,1]);
    if do_print
        print('-djpeg','updates')
    end
    
    % Final comparison estimated vs desired model
    figure(8); clf;
    for k=[11]
        plot(Stats(k,:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    plot(E_ij,'k','linewidth',2);
    legend({'with estimated model','desired'})
    axis([0,length(E_ij),0,1]);
    if do_print
        print('-djpeg','final')
    end
    
    % Progression of the estimation of J_ with gradient ascent
    figure(9)
    J_track_new = [.5];
    J_track_new = [J_track_new, J_track]; % plot from the iteration 0
    plot([0:11],J_track_new);
    xlabel('Iteration','fontsize',16);
    ylabel('Estimation of J_', 'fontsize', 16);
    xlim([0,11]);
    ylim([0.6,0.7]);
    if do_print
        print('-djpeg','J')
    end
end
