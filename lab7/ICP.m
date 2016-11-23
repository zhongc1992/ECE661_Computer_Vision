function Q_trans = ICP(cloud_1,cloud_2,iter);
row = 424;
col = 512;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ICP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1, find closest Euclidean distance pairs, if depth = 0, ignore the
%point directly
P_map = [];
Q_map = [];
for i = 1:1:row
    for j = 1:1:col
        temp_mat = ones(row,col);
        p = cloud_1{i,j};
        if (p == [0;0;0]) 
        else
            for k = 1:1:row
                for h = 1:1:col
                    q = cloud_2{k,h};
                    if (q == [0;0;0]) %if its depth = 0, do nothing
                    else
                        eucDis = sqrt(sum((p - q).^2));
                        if eucDis <= 0.1
                            temp_mat(k,h) = eucDis;
                        end
                    end
                end
            end
            [loc_r,loc_c] = find(temp_mat == min(min(temp_mat)));
            if min(min(temp_mat)) ~= 1
                P_map = [P_map p];
                Q_map = [Q_map cloud_2{loc_r(1),loc_c(1)}];
            end
            
        end
    end
end
filename1 = ['P_map_' num2str(iter)];
filename2 = ['Q_map_' num2str(iter)];
save(filename1,'P_map');
save(filename2,'Q_map');

%Estimate rotaion and translation matrices
N = max(size(P_map));
P_cen = sum(P_map,2)/N;
Q_cen = sum(Q_map,2)/N;

for i = 1:1:max(size(P_map))
    Mp(:,i) = P_map(:,i) - P_cen;
    Mq(:,i) = Q_map(:,i) - Q_cen;
end
C = Mq * Mp';
[U S V] = svd(C);
R_mat = V * U';
t_mat = P_cen - R_mat * Q_cen;
TransMat = [R_mat t_mat;0 0 0 1];
filename3 =  ['transmat_' num2str(iter)];
save(filename3,'TransMat');

%convert Q point cloud into homogenous
Q_trans = cell(row,col);
for i = 1:1:row
    for j = 1:1:col
        target = cloud_2{i,j};
        if target == [0;0;0]
            Q_trans{i,j} = [0;0;0];
        else
            target = [target;1]; %convert to homogenous with [x1;x2;x3;1]
            output_pt = TransMat * target;
            output_pt = output_pt./output_pt(4);
            Q_trans{i,j} = [output_pt(1);output_pt(2);output_pt(3)];
        end
    end
end
