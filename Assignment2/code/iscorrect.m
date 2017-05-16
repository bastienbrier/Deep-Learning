function res = iscorrect(state,lenbar)
part_1 = state(:,1:lenbar);
part_2 = state(:,lenbar + [1:lenbar]);

same  = all(part_1==part_2,2);
left  = all(part_1==part_2(:,[2:lenbar,1]),2);
right = all(part_1==part_2(:,[lenbar,1:lenbar-1]),2);


res = max(max(same,left),right);