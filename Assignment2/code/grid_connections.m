function Js  = grid_connections(sv,sh,J)
%% matrix containing all possible connections
Js = zeros(sv*sh,sv*sh);

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
