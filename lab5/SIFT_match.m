function [match,F1,D1,F2,D2]= SIFT_match(img1,img2)%return the matched pairs and feature points

% img1 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1048.JPG'); %read images
% img2 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1049.JPG');
% 
img1_gray = rgb2gray(img1);%convert image from rgb to gray
img2_gray = rgb2gray(img2);
input1 = double(img1_gray);%convert unit8 into double
input2 = double(img2_gray);


img1_size = size(input1);%calculate image size
row1 = img1_size(1);
col1 = img1_size(2);
img2_size = size(input2);
row2 = img2_size(1);
col2 = img2_size(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Using vl_sift function to compute sift feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[F1,D1] = vl_sift(single(input1));%F is the keypoints, D is the descriptor
[F2,D2] = vl_sift(single(input2));
D1 = double(D1);
D2 = double(D2);
location1 = cell(1,max(size(F1)));
location2 = cell(1,max(size(F2)));

for i = 1:1:max(size(F1))
    location1{1,i} = [round(F1(2,i)),round(F1(1,i))];
end

for i = 1:1:max(size(F2))
    location2{1,i} = [round(F2(2,i)),round(F2(1,i))];
end
%haar_scale = 2.2;%set the scale factor of haar filter
%haar_size = ceil(ceil((4*haar_scale))/2)*2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialize parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cor1_sum = max(size(F1));
cor2_sum = max(size(F2));
SSD = zeros(cor1_sum,cor2_sum);


%create new image
EUC = zeros(cor1_sum,cor2_sum);
EUC_display(1:(max(row1,row2)),1:col1+col2,1:3) = ...
zeros(max(row1,row2), col1+col2,3);
EUC_display(1:row1,1:col1,:) = img1;
EUC_display(1:row2,1+col1:col1+col2,:) = img2;
EUC_display = uint8(EUC_display);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euclidean distance -- directly compare the feature vectors of interest
%point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Start SIFT euclidean distance');
cccount = 0;
for i = 1:1:cor1_sum
    for j = 1:1:cor2_sum
        SSD(i,j) = sum((D1(:,i) - D2(:,j)).^2);
    end   
end
EUC = SSD.^0.5;
%Set threshold and ratio parameters of euclidean distance 
thresh_EUC = 5*abs(min(min(EUC)));
ratio_EUC = 0.85;
%figure(1)
%image(EUC_display);
%hold on;
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        if (EUC(i,j) < thresh_EUC && EUC(i,j) == min(EUC(i,:))) %also pick SSD(i,j) to be the min, otherwise there will be too much candidates
            temp_vec = sort(EUC(i,:));%Extract the first and second minimum
            first_min = temp_vec(1);
            second_min = temp_vec(2);
            if ((first_min/second_min) < ratio_EUC)
                cccount = cccount + 1;
                match(cccount,1) = ix;%store the matched pair's coordinates in match matrix
                match(cccount,2) = iy;%first two cols, coords of img 1
                match(cccount,3) = jx;
                match(cccount,4) = jy;%second two cols, coords of img 2
               % plot([iy;col1+jy],[ix;jx],'y-o','MarkerEdgeColor','y');
            end
        end
    end   
end
title('EUC correspondence matching plot');
disp('Finish SIFT EUC');
