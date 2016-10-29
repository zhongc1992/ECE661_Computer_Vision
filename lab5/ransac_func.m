function [inlier,outlier,homo] = ransac_func(pts,Eps,P,choN,distT)
%%% pts ---- Input corresponses points, row is x, col is y
%   pts(1,:) -- row of image 1
%   pts(2,:) -- col of image 1
%   pts(3,:) -- row of image 2
%   pts(4,:) -- col of image 2

%%% iterN ---- Number of iterations
%%% choN ---- Number of correspondences chosen at each trial
%%% distT ---- Decision threshold to construct the inlier set
%%% M ---- A minimum value for the size of inlier set to be accepted.
%%% Eps ---- Probability that any single correspondence is a false inlier, set to
%%%0.1 according to lecture note
%%% P ---- Probability that at least one of the N trials will be free of outliers
%%%set to 0.99 according to lecture note


M = round(max(size(pts)) * (1-Eps));
iterN = round(log(1-P)/log((1-(1-Eps)^choN))); 


all_inlier =cell(iterN,1);%Use cell to save reuslt from each iteration
all_outlier = cell(iterN,1);
all_homo = cell(iterN,1);

%totally N iteration
for ind = 1:1:iterN 
        inlier = [];
        outlier = [];
        %Randomly choose choN points from correspondence
        randInd = randperm(max(size(pts)));%random index in size of pts
        selePts = pts(randInd(1:choN),:);
        pts_1 = selePts(:,1:2);%correspondence points location from image 1
        pts_2 = selePts(:,3:4);%correspondence points location from image 2

        %Calculate homography from linear least square method
        A = [];%buile the matrix A, the matrix on the left_hand side
        for i = 1:1:choN
            A = [A;0,0,0,-1*pts_1(i,1),-1*pts_1(i,2),-1,pts_2(i,2)*pts_1(i,1),pts_2(i,2)*pts_1(i,2),pts_2(i,2)*1;1*pts_1(i,1),1*pts_1(i,2),1*1,0,0,0,-pts_2(i,1)*pts_1(i,1),- pts_2(i,1)*pts_1(i,2),-pts_2(i,1)*1];
        end
        [V D] = eig(A'*A);
        H = [V(1:3,1)' ; V(4:6,1)' ; V(7:9,1)'];
        all_homo{ind,1} = H;
        %Compare the estimated result with true homography
        all_pts_1 = pts(:,1:2);%all points from img1 of SIFT
        all_pts_2 = pts(:,3:4);%all points from img2 of SIFT
        tmp_pts_1 = all_pts_1;
        tmp_pts_1(:,3) = 1;
        for i = 1:1:max(size(all_pts_1))
            pt = H * tmp_pts_1(i,:)';
            true_pts_2(i,1) = pt(1)/pt(3);
            true_pts_2(i,2) = pt(2)/pt(3);
        end

        error_temp = (true_pts_2 - all_pts_2).^2;
        error_mat = (error_temp(:,1) + error_temp(:,2)).^0.5;%compute the error distance

        %compare each error with threshold, and conclude them as inlier or outlier
        for i = 1:1:max(size(all_pts_1))
            if (error_mat(i) < distT)
                inlier = [inlier;pts(i,:)];  
            else
                outlier = [outlier;pts(i,:)];
            end
        end

        all_inlier{ind,1} = inlier;
        all_outlier{ind,1} = outlier;
end
      
max_num = 0;
mark = 0;
while (mark == 0)
    for i = 1:1:iterN
        pts_num = max(size(all_inlier{i,1}));
        if (pts_num > M) && (pts_num > max_num)
            max_num = pts_num;
            mark = i;
        end
    end
    if (mark == 0) %if nothing satisfied, reduce the threshold
        M = M - 10;
    end
end

inlier = all_inlier{mark,1};
outlier = all_outlier{mark,1};
homo = all_homo{mark,1};

end %end of the function