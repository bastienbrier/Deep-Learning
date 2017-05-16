function states = get_all_states(nnodes)
    % nnodes: number of node
    % states: returns all possible states
    
states = [1; -1]; % original states

% compute all possible states
for i = 2:nnodes
  s = size(states,1);
  states = [[ones(s,1) states]; [-1*ones(s,1) states]];
end