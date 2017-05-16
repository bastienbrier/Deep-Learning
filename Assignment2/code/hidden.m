
randn('seed',0);
rand('seed',0);

sz_half            = 4;
[bars,direction]   = genbars(500,sz_half);
[nsamples,n_input] = size(bars);

if do_print
    k      = 3;
    figure
    showbar(bars(1:100,:));
end

randn('seed',100);
n_hidden    = 12;
%% input-to-input connections
W_i2i       = make_proper(0.1*randn(n_input,n_input));
W_i2h       =             0.1*randn(n_input,n_hidden);
W_h2h       = make_proper(0.1*randn(n_hidden,n_hidden));

meths = {'ising','rbm','bm'};

for m_i = [1] %, %1:length(meths)]
    meth = meths{m_i};
    switch meth
        case 'ising'
            states          = get_all_states(n_input);
            W_all           = W_i2i;
        case 'bm'
            states          = get_all_states(n_input + n_hidden);
            states_hidden   = get_all_states(n_hidden);
            W_all           = [[W_i2i,W_i2h];[W_i2h',W_h2h]];
        case {'rbm','rbm_mcmc'}
            states          = get_all_states(n_input + n_hidden);
            states_hidden   = get_all_states(n_hidden);
            W_all           = [[0*W_i2i,W_i2h];[W_i2h',0*W_h2h]];
        
    end
    cnt_tot = 0;
    P_all   = zeros(100*10,size(states,1));
    ok      = iscorrect(states,sz_half)';
    
    for it_out = [1:100]
        fprintf(2,'\nit %i',it_out);
        nits_gd     = 40;
        ener_all    = sum((states.*(states*W_all)),2);
        Z           = sum(exp(-ener_all));
        
        %P       = exp(-ener)/sum(exp(-ener));
        % \sum_n < X_i,X_j> P(h|x(n))
        
        E_data =  0;
        for n = 1:nsamples
            state_o = bars(n,:);
            switch meth
                case 'ising'
                E_data  = E_data + state_o'*state_o/nsamples;
                ener    = sum((state_o.*(state_o*W_all)),2);
                case {'bm','rbm'}
                n_h     = size(states_hidden,1);
                state_j = [repmat(state_o,[n_h,1]),states_hidden];
                ener    = sum((state_j.*(state_j*W_all)),2);
                P_n     = exp(-ener)/sum(exp(-ener));
                E_data  = E_data + (state_j'*bsxfun(@times,P_n,state_j))/nsamples;
                case 'rbm_mcmc'
                    
            end
            Ps(it_out,n)  = sum(exp(-ener)/Z);
        end
        
        for it_gd = [1:nits_gd]
            switch meth
                case 'rbm_mcmc'
                otherwise
                    %% brute force summation
                    ener    = sum((states.*(states*W_all)),2);
                    P       = exp(-ener)/sum(exp(-ener));
                    E_ij    = states'*bsxfun(@times,P,states);
            end
            
            grad    = (E_data - E_ij);
            %% make sure self-connections stay zero.
            grad    =  grad - diag(diag(grad));
            %% make sure the connections stay symmetric
            grad    = (grad + grad')/2;
            
            %% make sure the RBM separates observable and hidden nodes
            if strcmp(meth,'rbm')||(strcmp(meth,'rbm_mcmc'))
                grad(W_all==0) = 0;
            end
            W_all   = W_all - .05*grad;
            fprintf(2,'.');
            cnt_tot         = cnt_tot + 1;
            
            %% this should be going down in each iteration
            diffs(cnt_tot)  = sum((E_ij(:) - E_data(:)).^2);
            
            %% this measures the network's precision 
            P_ok(cnt_tot)   = sum(P'.*ok);
        end
    end
    info = struct('P_ok',P_ok,'KL_p',KL_p,'Ps',Ps,'diffs',diffs);
    switch meth
        case 'ising'
            info_ising  = info;
        case 'rbm'
            info_rbm    = info;
        case 'bm'
            info_bm     = info;
    end
end

figure,
plot(sum(log(info_ising.Ps),2),'k');
hold on
plot(sum(log(info_bm.Ps),2),'r');
hold on,
plot(sum(log(info_rbm.Ps),2),'g');
legend('ising','bm','rbm');
xlabel('Gradient ascent iteration');
ylabel('log likelihood')


figure,
plot(info_ising.P_ok);
hold on
plot(info_bm.P_ok,'r');
hold on,
plot(info_rbm.P_ok,'g');


%[m,i_max] = max(P);
%P       = perf_c{2}.P;
idxs    = randsample(1:length(P),100,'true',P);
figure
showbar(states(idxs(1:100),[1:n_input]));



[m,i_max] = min(P);
break;
nits_gibbs  = 2e4;
it_burn     = 1e3;
rng = [-1,1];
figure,
%subplot(1,2,1);
showbar(states(i_max,[1:n_input]));
imshow(reshape(states(i_max,:),[sv,sh]),rng)
subplot(1,2,2);
imshow(reshape(vals_all(i_min,:),[sv,sh]),rng)
%idxs = randsample(P_al
%W_i2i_exact = W_i2i;
figure,plot(E_data(:)); hold on,plot(E_ij(:),'r'); fprintf(2,'\n')
figure,plot(diffs);
%figure,hinton(W_i2i_exact);

clear distStat E_track
for it_gd = [1:nits_gd]
    if it_gd==1
        X_init = 2*(rand(1,n_input)>.5)-1;
        X      = X_init;
    end
    sum_model_prod  = 0;
    nits_prod       = 0;
    for it_gibbs =1:nits_gibbs
        coord = ceil(rand*n_input);
        weights     = W_i2i(coord,:);
        ener_plus   =  sum(weights.*X);
        ener_minus  = -sum(weights.*X);
        posterior   = exp(-ener_plus)./(exp(-ener_plus) + exp(-ener_minus));
        is_above    = double(posterior>rand);
        X(coord)    = +1*is_above + (-1)*(1-is_above);
        if it_gibbs>it_burn
            if (mod(it_gibbs,10)==0)
                sum_model_prod = sum_model_prod + X'*X;
                nits_prod       = nits_prod  + 1;
            end
        end
    end
    if mod(it_gd,10)==0;
        fprintf(2,'-')
    end
    E_model             = sum_model_prod/nits_prod;
    distStat(it_gd)     = sum((E_model(:) - E_data(:)).^2);
    E_track(it_gd,:)    = E_model(:);
    grad                = (E_data - E_model);
    grad                = make_proper(grad);
    W_i2i               = W_i2i - .01*grad;
end

figure,plot(distStat);
hold on,plot(diffs,'r');

figure,plot(E_model(:));
hold on,plot(E_data(:),'r');

