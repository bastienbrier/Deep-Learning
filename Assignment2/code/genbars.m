function [bars,direction] = genbars(nbars,lenbar)
rand('seed',0);
for k=1:nbars
    bar_top(k,:) = 2*(rand(1,lenbar) > .5)-1;
    direction(k) = randsample(3,1);
    
    %% shift by {-1,0,1} (with cyclic permutation)
    switch direction(k)
        case 1
            bar_bot(k,:) = bar_top(k,:);
        case 2
            bar_bot(k,:) = bar_top(k,[2:end,1]);
        case 3
            bar_bot(k,:) = bar_top(k,[end,1:end-1]);
    end
end
bars = [bar_top,bar_bot];