clc;
clear;
 
%img1 = imread('E:\2016Fall\661\HW4\pair3\1.jpg'); %read images
%img2 = imread('E:\2016Fall\661\HW4\pair3\2.jpg');
img1 = imread('E:\2016Fall\661\HW4\IMG_1038.jpg'); %read images
img2 = imread('E:\2016Fall\661\HW4\IMG_1039.jpg');
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
 
haar_scale = 2.2;%set the scale factor of haar filter
haar_size = ceil(ceil((4*haar_scale))/2)*2;%size of haar wavelet matrices
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize haar wavelet matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hx = ones(haar_size,haar_size);
Hy = ones(haar_size,haar_size);
Hx(:,1:haar_size/2) = -1;
Hy(haar_size/2 + 1:end,:) = -1;
%Gaussian smooth
smooth_filter = fspecial('gaussian', 5*haar_scale, haar_scale);
input1 = imfilter(input1,smooth_filter);
input2 = imfilter(input2,smooth_filter);
%Process of filtering
input1_x = imfilter(input1,Hx);
input1_y = imfilter(input1,Hy);
input2_x = imfilter(input2,Hx);
input2_y = imfilter(input2,Hy);
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot graphs after haar filtering 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
subplot(2,2,1)
image(input1_x);
colormap(gray(256));
title('Gradient in X direction of image1');
subplot(2,2,2)
image(input1_y);
title('Gradient in Y direction of image1');
subplot(2,2,3)
image(input2_x);
title('Gradient in X direction of image2');
subplot(2,2,4)
image(input2_y);
title('Gradient in Y direction of image2');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Construct the C matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C_1 = build_C_mat(input1_x,input1_y,haar_scale);
C_2 = build_C_mat(input2_x,input2_y,haar_scale);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Check corners through rank of each element in matrix C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
corner_map1 = check_rank(C_1); %corner map extracted from matrix C
corner_map2 = check_rank(C_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Corner responses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
response_1 = corner_response(C_1,corner_map1);
response_2 = corner_response(C_2,corner_map2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get final interest points from harris corner detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[harris_point1,location1] = non_max_suppress(response_1,response_2);
[harris_point2,location2] = non_max_suppress(response_2,response_1);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SSD 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cor1_sum = sum(sum(harris_point1));
cor2_sum = sum(sum(harris_point2));
SSD_win = haar_scale * 20; %define the window size to use
SSD = zeros(cor1_sum,cor2_sum);
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        for p = -SSD_win/2:1:SSD_win/2
            for q = -SSD_win/2:1:SSD_win/2
                if((ix+p>0)&&(ix+p<row1)&&(iy+q>0)&&(iy+q<col1)&&(jx+p>0)&&(jx+p<row2)&&(jy+q)>0&&(jy+q)<col2)
                 SSD(i,j) = SSD(i,j) + abs(input1(ix+p,iy+q) - input2(jx+p,jy+q))^2;
                end
            end
        end
    end   
end
 
%create new image
SSD_display(1:(max(row1,row2)),1:col1+col2,1:3) = ...
zeros(max(row1,row2), col1+col2,3);
SSD_display(1:row1,1:col1,:) = img1;
SSD_display(1:row2,1+col1:col1+col2,:) = img2;
SSD_display = uint8(SSD_display);
figure(3)
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
                plot([iy;col1+jy],[ix;jx],'r-o','MarkerEdgeColor','r');
            end
        end
    end   
end
title('SSD correspondence matching plot');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NCC 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NCC_win = haar_scale * 20; %define the window size to use,same as SSD_win
NCC = zeros(cor1_sum,cor2_sum);
input1_clone = zeros(row1 + NCC_win,col1 + NCC_win);
input1_clone(1+NCC_win/2:1:row1+NCC_win/2,1+NCC_win/2:1:col1+NCC_win/2) = input1(1:1:row1,1:1:col1);
input2_clone = zeros(row2 + NCC_win,col2 + NCC_win);
input2_clone(1+NCC_win/2:1:row2+NCC_win/2,1+NCC_win/2:1:col2+NCC_win/2) = input2(1:1:row2,1:1:col2);
for i = 1:1:cor1_sum
    ix = location1{1,i}(1);
    iy = location1{1,i}(2);
    for j = 1:1:cor2_sum
        jx = location2{1,j}(1);
        jy = location2{1,j}(2);
        numerator = 0;
        denominator_A = 0;
        denominator_B = 0;
        mean1 = mean(mean(input1_clone(ix:1:ix+NCC_win,iy:1:iy+NCC_win)));
        mean2 = mean(mean(input2_clone(jx:1:jx+NCC_win,jy:1:jy+NCC_win)));
        for p = -NCC_win/2:1:NCC_win/2
            for q = -NCC_win/2:1:NCC_win/2 
                if((ix+p>0)&&(ix+p<row1)&&(iy+q>0)&&(iy+q<col1)&&(jx+p>0)&&(jx+p<row2)&&(jy+q>0)&&(jy+q<col2))
                     numerator = numerator + (input1(ix+p,iy+q)-mean1)*(input2(jx+p,jy+q)-mean2);
                     denominator_A = denominator_A + (input1(ix+p,iy+q)-mean1)^2;
                     denominator_B = denominator_B + (input2(jx+p,jy+q)-mean2)^2;
                end
            end
        end
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