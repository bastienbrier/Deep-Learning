%% Parameters to choose
rand('seed',0);
do_viz              = 0;
sz_half             = 4;
debug               = 0;
diagnostic          = 1;
bars_sampled        = 0;
do_save             = 1;

%% # of gradient descent operations and part selection
it_max  = 2000;
part    = 1; % from [1,2,3]
n_hidden_wt = 8;

[bars,direction]    = genbars(500,sz_half);
[nsamples,n_input]  = size(bars);

if do_viz
    figure,showbar(bars(1:256,:));
end

methods = {'ising','bm','rbm','rbm-cd-1','rbm-cd-10','rbm-cd-20','rbm-cd-50','rbm-cd-100'};

%% function
%for part = [1,2]
    if part == 1
        meths_wt    = [1:3];
    elseif part == 2
        meths_wt    = [3:5];
    elseif part == 3
        meths_wt    = [3:8];
    end
    for n_hidden = n_hidden_wt
        randn('seed',0);
        %% random initialization of network connections
        %% input-to-input
        W_i2i       = make_proper(0.1*randn(n_input,n_input));
        %% input-to-hidden
        W_i2h       =             0.1*randn(n_input,n_hidden);
        %% hidden-to-hidden
        W_h2h       = make_proper(0.1*randn(n_hidden,n_hidden));
        
        %% set diagnostic=0;  for larger models (optional part of assignment)
        if diagnostic
            states_input  = get_all_states(n_input);
            states_hidden = get_all_states(n_hidden);
            states_joint  = get_all_states(n_input + n_hidden);
        else
            states_input  = [];
            states_hidden = [];
            states_joint  = [];
        end
        
        for m_i = meths_wt
            meth = methods{m_i};
            fprintf(' Method: %s',meth);
            
            switch meth
                case 'ising'
                    states          = states_input;
                    W_all           = W_i2i;
                    step_gd         = .1;
                case 'bm'
                    states          = states_joint;
                    W_all           = [[W_i2i,W_i2h];[W_i2h',W_h2h]];
                    step_gd         = .1;
                case {'rbm','rbm-cd-1','rbm-cd-10','rbm-cd-20','rbm-cd-50','rbm-cd-100'}
                    states          = states_joint;
                    W_all           = [[zeros(n_input),W_i2h];[W_i2h',zeros(n_hidden)]];
                    if strcmp(meth,'rbm')
                        step_gd = .1;
                    elseif strcmp(meth,'rbm-cd-1') && part == 2
                        step_gd = .015;
                    else
                        %% we have to decrease the update step for CD
                        %% or else the training diverges
                        step_gd = .02;
                    end
            end
            
            if diagnostic
                %% indicates whether observable states satisfy `shifting' constraint
                ok      = iscorrect(states,sz_half)';
            end
            
            %% likelihood of every sample for all gradient descent iterations
            Ps          = zeros(it_max,nsamples);
            
            for it  = [1:it_max]
                
                if mod(it,20) ==1,    fprintf(2,'\n it %i\t',it);  end
                left_out = 0;
                if diagnostic
                    %% update the partition function
                    %% energy of all network states
                    ener_all    = (-1/2)*sum((states.*(states*W_all)),2);
                    %% partition function
                    Z           = sum(exp(-ener_all));
                    %% probability of any state
                    P           = exp(-ener_all)/Z;
                end
                
                %% 'awake' phase computes
                %% 1/n [\sum_n < Y_i,Y_j>_{P(h|x(n))}]
                %% 'dream' phase computes
                %%  < Y_i,Y_j>_{P(h,x)}
                
                E_awake  = 0;
                E_dream = 0;
                
                for m = 1:nsamples
                    state_o = bars(m,:);
                    switch meth
                        case 'ising'
                            %% energy of observed state
                            ener_clamped = (-1/2)*state_o*W_all*state_o';
                            %% contribution of m-th sample to 'awake' matrix
                            E_positive   = (exp(-ener_clamped) / sum(exp(-ener_clamped))) * (state_o' * state_o);
                            
                            E_awake      = E_awake + E_positive/nsamples;
                        case {'bm','rbm'}
                            %% this matrix contains the combination of the
                            %% observed state state_o
                            %% with all possible hidden variable states,
                            %% e.g. for M =4
                            %% [state_o , -1, -1 ,-1 ,-1 ]
                            %% [state_o , -1, -1, -1,  1 ]
                            %% ...
                            %% [state_o ,  1,  1,  1,  1 ]
                            
                            state_clamped   = [repmat(state_o,[size(states_hidden,1),1]),states_hidden];
                            ener_clamped    = (-1/2)*sum((state_clamped.*(state_clamped*W_all)),2);
                            %% P(h|state_o): store as an 2^M vector,
                            %% M being the #of hidden variables
                            P_n             = exp(-ener_clamped) / sum(exp(-ener_clamped));
                            E_awake         = E_awake + (state_clamped'*bsxfun(@times,P_n,state_clamped))/nsamples;
                        case {'rbm-cd-1','rbm-cd-10','rbm-cd-20','rbm-cd-50','rbm-cd-100'}
                            if m==1
                                %% update the observed-to-hidden connections
                                W_inter = W_all([1:n_input],n_input+[1:n_hidden]);
                            end
                            if diagnostic
                                state_j         = [repmat(state_o,[size(states_hidden,1),1]),states_hidden];
                                ener_clamped    = (-1/2)*sum((state_j.*(state_j*W_all)),2);
                                P_n             = exp(-ener_clamped) / Z;
                            end
                            
                            %% h: hidden, v: visible
                            %% positive phase: P(h|v),  [M x 1] vector of posteriors
                            P_hidden_0  = 1 ./ (1+exp(-2*state_o*W_inter)); %(28) in theory
                            debug  = 0;
                            if debug
                                %% this could be useful for debugging:
                                %% verify that P(h = h_k|v)  (h_k= states_hidden(k,:))
                                %% evalutes to the same quantity, whether evaluated by
                                %% (i)  brute force summation 'diagnostic'
                                %% P_k = exp(-E(h_k,v))/sum_k' exp(- E(h_k',v))
                                %% or based on the factorization of P(h|v):
                                %% P_k = P(h^1 = h_k^1|v) * P(h^2 = h_k^2|v) * ..
                                for k=1:length(P_n)
                                    state_          = states_hidden(k,:);
                                    P_s(k)          = prod(P_hidden_0.^(state_==1).*(1-P_hidden_0).^(state_==-1));
                                end
                                figure,plot(P_n); hold on,plot(P_s,'r');
                                legend({'brute force','factorized'});
                            end
                            %% 'awake' phase
                            %% <h_i v_j>_{P(h|v^i)} = sum_{label_h = -1/1} label_h * P(h_i = label_h|v) *  v_j
                            E_positive  = state_o'*(2*P_hidden_0-1)*n_hidden; %(36) in theory
                            E_awake      = E_awake + ...
                                [zeros(n_input), E_positive;...
                                E_positive',zeros(n_hidden)]/nsamples;
                            
                            obs_K = state_o;
                            %% 'dream' phase with Contrastive Divergence (read assignment for details)
                            %% <h_i v_j>_{P(h,v)} ~= sum_{label_h = -1/1} label_h * P(h_i = label_h|v^K) *  v^K_j
                            switch meth
                                case 'rbm-cd-1'
                                    K = 1;
                                case 'rbm-cd-10'
                                    K = 10;
                                case 'rbm-cd-20'
                                    K = 20;
                                case 'rbm-cd-50'
                                    K = 50;
                                case 'rbm-cd-100'
                                    K = 100;
                            end
                            
                            [obs_K,P_hidden_K]  = block_gibbs(obs_K,W_inter,K);
                            E_negative          = obs_K'*(2*P_hidden_K-1)*n_hidden;
                            
                            E_dream             = E_dream + [zeros(n_input), E_negative;...
                                E_negative',zeros(n_hidden)]/(nsamples);
                    end
                    if diagnostic
                        
                        %% P(x^m;W_{it}) =  sum_h P(x^m,h;W_{it})
                        Ps(it,m) = sum(exp(-ener_clamped))/Z;
                    end
                end
                Zs(it)  = Z;
                
                if ismember(meth,{'bm','ising','rbm'})
                    %% brute force summation for 'dream' part
                    E_dream = states'*bsxfun(@times,P,states);
                end
                
                grad    = (E_awake - E_dream);
                W_all   = W_all + step_gd*grad;
                
                %% make sure the RBM separates observable and hidden nodes
                if strcmp(meth,'rbm')||(strcmp(meth,'rbm-cd'))
                    W_all(1:n_input,1:n_input) = 0;
                    W_all(n_input + [1:n_hidden],n_input + [1:n_hidden]) = 0;
                end
                %% make sure the connections stay symmetric
                W_all = (W_all + W_all')/2;
                %% make sure self-connections stay zero.
                W_all = W_all - diag(diag(W_all));
                
                fprintf(2,'.');
                
                %% this should be going down in each iteration
                diffs(it)  = sum((E_dream(:) - E_awake(:)).^2);
                if diagnostic
                    %% this measures the network's precision
                    P_ok(it)   = sum(P'.*ok);
                else
                    P_ok = [];
                end
            end
            fprintf(2,'\n');
            info = struct('P_ok',P_ok,'Ps',Ps,'diffs',diffs);
            switch meth
                case 'ising'
                    info_ising      = info;
                case 'rbm'
                    info_rbm        = info;
                case 'bm'
                    info_bm         = info;
                case 'rbm-cd-1'
                    info_rbm_cd_1   = info;
                case 'rbm-cd-10'
                    info_rbm_cd_10  = info;
                case 'rbm-cd-20'
                    info_rbm_cd_20  = info;
                case 'rbm-cd-50'
                    info_rbm_cd_50  = info;
                case 'rbm-cd-100'
                    info_rbm_cd_100  = info;
            end
            
            %% replace 0 with 1 if you want to some bars sampled from your model
            if bars_sampled == 1
                idxs    = randsample(1:length(P),100,'true',P);
                figure
                showbar(states(idxs(1:100),[1:n_input]));
                if do_save
                    print('-djpeg');
                end
            end
        end
        
        %for part = [1,2]
        %% generate plots
        if part == 1
            info_1 = info_ising;
            info_2 = info_bm;
            info_3 = info_rbm;
            legend_strings = {'ising','bm','rbm'};
            fnm_string  = sprintf('brute_force_nhidden_%i',n_hidden);
        elseif part == 2
            info_1 = info_rbm;
            info_2 = info_rbm_cd_1;
            info_3 = info_rbm_cd_10;
            legend_strings = {'rbm, brute force','rbm, CD-1','rbm, CD-10'};
            fnm_string  = sprintf('monte_carlo_nhidden_%i',n_hidden);
        elseif part == 3
            info_1 = info_rbm;
            info_2 = info_rbm_cd_1;
            info_3 = info_rbm_cd_10;
            info_4 = info_rbm_cd_20;
            info_5 = info_rbm_cd_50;
            info_6 = info_rbm_cd_100;
            legend_strings = {'rbm, brute force','rbm, CD-1','rbm, CD-10','rbm, CD-20','rbm, CD-50','rbm, CD-100'};
            fnm_string  = sprintf('monte_carlo_nhidden_%i',n_hidden);
        end
        
        if part == 1 || part == 2
            figure, ha = axes;
            plot(sum(log(info_1.Ps),2),'k','linewidth',2);
            hold on
            plot(sum(log(info_2.Ps),2),'r','linewidth',2);
            hold on,
            plot(sum(log(info_3.Ps),2),'g','linewidth',2);
            legend(legend_strings,'location','southeast','fontsize',20);
            set(ha,'fontsize',20);
        elseif part == 3
            figure, ha = axes;
            plot(sum(log(info_1.Ps),2),'k','linewidth',1);
            hold on
            plot(sum(log(info_2.Ps),2),'r','linewidth',1);
            hold on,
            plot(sum(log(info_3.Ps),2),'g','linewidth',1);
            hold on,
            plot(sum(log(info_4.Ps),2),'b','linewidth',1);
            hold on,
            plot(sum(log(info_5.Ps),2),'m','linewidth',1);
            hold on,
            plot(sum(log(info_6.Ps),2),'c','linewidth',1);
            legend(legend_strings,'location','southeast','fontsize',10);
            set(ha,'fontsize',20);
        end

        xlabel('Gradient ascent iteration','fontsize',20);
        ylabel('log likelihood','fontsize',20)
        fnm_im = ['log_likelihood_',fnm_string];
        if do_save
            print('-djpeg',fnm_im);
        end
        %end
    end
%end

