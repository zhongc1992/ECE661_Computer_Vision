clc;
clear;
% 
img1 = imread('E:\2016Fall\661\HW4\IMG_1038.jpg'); %read images
img2 = imread('E:\2016Fall\661\HW4\IMG_1039.jpg');
%img1 = imread('E:\2016Fall\661\HW4\pair1\1.jpg'); %read images
%img2 = imread('E:\2016Fall\661\HW4\pair1\2.jpg');
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
figure(1)
image(EUC_display);
hold on;
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
                plot([iy;col1+jy],[ix;jx],'y-o','MarkerEdgeColor','y');
            end
        end
    end   
end
title('EUC correspondence matching plot');
disp('Finish SIFT EUC');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SSD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%create new image for SSD
SSD_display(1:(max(row1,row2)),1:col1+col2,1:3) = ...
zeros(max(row1,row2), col1+col2,3);
SSD_display(1:row1,1:col1,:) = img1;
SSD_display(1:row2,1+col1:col1+col2,:) = img2;
SSD_display = uint8(SSD_display);
figure(2)
image(SSD_display);
hold on;
 
thresh_SSD = 5*abs(min(min(SSD)));
ratio_SSD = 0.85;
count_SSD = 0;
count_sub_SSD = 0;
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        if (SSD(i,j) < thresh_SSD)
            temp_vec = sort(SSD(i,:));%Extract the first and second minimum
            first_min = temp_vec(1);
            second_min = temp_vec(2);
           
            if ((first_min/second_min) < ratio_SSD)
                count_SSD =count_SSD + 1;
                rand_color = rand(1, 3);
                plot([iy;col1+jy],[ix;jx],'r-o','MarkerEdgeColor','r');%,'Color',rand_color(1,:));
            end
        end
    end   
end
title('SSD correspondence matching plot');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NCC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NCC = zeros(cor1_sum,cor2_sum);
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        mean1 = mean(D1(:,i));
        mean2 = mean(D2(:,j));
        numerator = sum((D1(:,i)-mean1).*(D2(:,j)-mean2));
        denominator_A = sum((D1(:,i)-mean1).^2);
        denominator_B = sum((D2(:,j)-mean2).^2);
        NCC(i,j) = numerator/(denominator_A * denominator_B)^0.5;
    end   
end

%create new image
NCC_display(1:(max(row1,row2)),1:col1+col2,1:3) = ...
zeros(max(row1,row2), col1+col2,3);
NCC_display(1:row1,1:col1,:) = img1;
NCC_display(1:row2,1+col1:col1+col2,:) = img2;
NCC_display = uint8(NCC_display);
figure(4)
image(NCC_display);
hold on;

thresh_NCC = 0.9*abs(max(max(NCC)));
ratio_NCC = 1.09;
count_NCC = 0;
count_NCC_sub = 0;
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        if (NCC(i,j) > 0)
            if (NCC(i,j) > thresh_NCC)
                temp_vec = sort(NCC(i,:));%Extract the first and second maximum
                first_max = temp_vec(end);
                second_max = temp_vec(end-1);
                count_NCC_sub = count_NCC_sub + 1;
                if ((first_max/second_max) > ratio_NCC)
                    count_NCC =count_NCC + 1;
                    rand_color = rand(1, 3);
                    plot([iy;col1+jy],[ix;jx],'b-x','MarkerEdgeColor','b');
                end
            end
        end
    end   
end
title('NCC correspondence matching plot');