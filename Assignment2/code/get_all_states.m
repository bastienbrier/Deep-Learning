
function states    = get_all_states(nnodes);
%% generate an exhaustive enumaration of all possible 
%% 2^nnodes
%% network states

states        = [];
for d = [1:nnodes]
    append   = ones(max(size(states,1),1),1);
    states = [[states,-append];[states,append]];
end
