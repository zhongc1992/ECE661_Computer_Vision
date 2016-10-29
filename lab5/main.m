clc;
clear;
flag_dis = 0;%flag to display homography between pairs

img1 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1048.JPG'); %read images
img2 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1049.JPG');
img3 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1050.JPG'); %read images
img4 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1051.JPG');
img5 = imread('E:\2016Fall\661\HW5\imgset1Down\IMG_1052.JPG'); %read images

img1_gray = rgb2gray(img1);%convert image from rgb to gray
img2_gray = rgb2gray(img2);
img3_gray = rgb2gray(img3);
img4_gray = rgb2gray(img4);
img5_gray = rgb2gray(img5);

input1 = double(img1_gray);%convert unit8 into double
input2 = double(img2_gray);
input3 = double(img3_gray);
input4 = double(img4_gray);
input5 = double(img5_gray);

img1_size = size(input1);%calculate image size
row1 = img1_size(1);
col1 = img1_size(2);
img2_size = size(input2);
row2 = img2_size(1);
col2 = img2_size(2);
img3_size = size(input3);
row3 = img3_size(1);
col3 = img3_size(2);
img4_size = size(input4);
row4 = img4_size(1);
col4 = img4_size(2);
img5_size = size(input5);
row5 = img5_size(1);
col5 = img5_size(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters for RANSAC
Eps = 0.1;%Probability that any single correspondence is a false inlier
P = 0.99; %Probability that at least one of the N trials will be free of outliers
choN = 6;% Number of correspondences chosen at each trial
distT = 3 * 0.8;% Decision threshold to construct the inlier set, commonly equals to a 3*a small number between 0.5 - 2

[match12,F1,D1,F2,D2] = SIFT_match(img1,img2);
[inlier12,outlier12,homo12] = ransac_func(match12,Eps,P,choN,distT);

[match23,F2,D2,F3,D3] = SIFT_match(img2,img3);
[inlier23,outlier23,homo23] = ransac_func(match23,Eps,P,choN,distT);

[match43,F4,D4,F3,D3] = SIFT_match(img4,img3);
[inlier43,outlier43,homo43] = ransac_func(match43,Eps,P,choN,distT);

[match54,F5,D5,F4,D4] = SIFT_match(img5,img4);
[inlier54,outlier54,homo54] = ransac_func(match54,Eps,P,choN,distT);

homo13 = homo12 * homo23;
homo53 = homo54 * homo43;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% draw line for corresponding points
if (flag_dis == 1)
    figure(1)
    display12(1:(max(row1,row2)),1:col1+col2,1:3) = ...
    zeros(max(row1,row2), col1+col2,3);
    display12(1:row1,1:col1,:) = img1;
    display12(1:row2,1+col1:col1+col2,:) = img2;
    display12 = uint8(display12);
    image(display12);
    hold on
    for i = 1:1:max(size(inlier12))
        plot([inlier12(i,2);inlier12(i,4)+col1],[inlier12(i,1);inlier12(i,3)],'y-o','MarkerEdgeColor','y')
    end
    for i = 1:1:max(size(outlier12))
        plot([outlier12(i,2);outlier12(i,4)+col1],[outlier12(i,1);outlier12(i,3)],'r-o','MarkerEdgeColor','r')
    end
    hold off

    figure(2)
    display23(1:(max(row1,row2)),1:col1+col2,1:3) = ...
    zeros(max(row1,row2), col1+col2,3);
    display23(1:row1,1:col1,:) = img2;
    display23(1:row2,1+col1:col1+col2,:) = img3;
    display23 = uint8(display23);
    image(display23);
    hold on
    for i = 1:1:max(size(inlier23))
        plot([inlier23(i,2);inlier23(i,4)+col1],[inlier23(i,1);inlier23(i,3)],'y-o','MarkerEdgeColor','y')
    end
    for i = 1:1:max(size(outlier12))
        plot([outlier23(i,2);outlier23(i,4)+col1],[outlier23(i,1);outlier23(i,3)],'r-o','MarkerEdgeColor','r')
    end
    hold off
    
    figure(3)
    display34(1:(max(row1,row2)),1:col1+col2,1:3) = ...
    zeros(max(row1,row2), col1+col2,3);
    display34(1:row1,1:col1,:) = img3;
    display34(1:row2,1+col1:col1+col2,:) = img4;
    display34 = uint8(display34);
    image(display34);
    hold on
    for i = 1:1:max(size(inlier43))
        plot([inlier43(i,4);inlier43(i,2)+col1],[inlier43(i,3);inlier43(i,1)],'y-o','MarkerEdgeColor','y')
    end
    for i = 1:1:max(size(outlier43))
        plot([outlier43(i,4);outlier43(i,2)+col1],[outlier43(i,3);outlier43(i,1)],'r-o','MarkerEdgeColor','r')
    end
    hold off
    
    figure(4)
    display45(1:(max(row1,row2)),1:col1+col2,1:3) = ...
    zeros(max(row1,row2), col1+col2,3);
    display45(1:row1,1:col1,:) = img4;
    display45(1:row2,1+col1:col1+col2,:) = img5;
    display45 = uint8(display45);
    image(display45);
    hold on
    for i = 1:1:max(size(inlier54))
        plot([inlier54(i,4);inlier54(i,2)+col1],[inlier54(i,3);inlier54(i,1)],'y-o','MarkerEdgeColor','y')
    end
    for i = 1:1:max(size(outlier54))
        plot([outlier54(i,4);outlier54(i,2)+col1],[outlier54(i,3);outlier54(i,1)],'r-o','MarkerEdgeColor','r')
    end
    hold off
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image mosaic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate the four corner points of the mosaic image,[x y 1] -- [row col
%1]
%%try to map img1 into img 3 and img 5 to img 3, their might exist negative coord, which is
%%acceptable in mosaic image

%calcu
TL13 = homo13*[1 1 1]'; %top left corner of img 1
TL13 = TL13/TL13(3);
BL13 = homo13*[row1 1 1]'; %bottom left corner of img 1
BL13 = BL13/BL13(3);
TR53 = homo53*[1 col1 1]'; %top right corner of img 5
TR53 = TR53/TR53(3);
BR53 = homo53*[row1 col1 1]';%bottom right corner of img 5
BR53 = BR53/BR53(3);
max_x = max([TL13(1) BL13(1) TR53(1) BR53(1)]); %find image boundary
max_y = max([TL13(2) BL13(2) TR53(2) BR53(2)]);
min_x = min([TL13(1) BL13(1) TR53(1) BR53(1)]);
min_y = min([TL13(2) BL13(2) TR53(2) BR53(2)]);
col_mosaic = max_y-min_y+1; %decide img row and col
row_mosaic = max_x-min_x+1;

img_mosaic1 = zeros(ceil(row_mosaic),ceil(col_mosaic),3);%container of the final image

img_mosaic1 = mosaic_func(img_mosaic1,img1,min_x,min_y,homo13);
img_mosaic1 = mosaic_func(img_mosaic1,img2,min_x,min_y,homo23);
img_mosaic1 = mosaic_func(img_mosaic1,img4,min_x,min_y,homo43);
img_mosaic1 = mosaic_func(img_mosaic1,img5,min_x,min_y,homo53);
img_mosaic1(2-min_x:row1+1-min_x, 2-min_y:col1+1-min_y,:) = img3;%insert image 3 to the mid at the last step 

figure(6)
imtool(uint8(img_mosaic1));
imwrite(img_mosaic1,'E:\2016Fall\661\HW5\mosaic.jpg');

%%%%%% Apply dogleg to refine solution
f_homo12 = dogleg_func(homo12,inlier12); 
f_homo23 = dogleg_func(homo23,inlier23); 
f_homo43 = dogleg_func(homo43,inlier43); 
f_homo54 = dogleg_func(homo54,inlier54); 
f_homo13 = f_homo12 * f_homo23;
f_homo53 = f_homo54 * f_homo43;

img_mosaic2 = zeros(ceil(row_mosaic),ceil(col_mosaic),3);%container of the final image
img_mosaic2 = mosaic_func(img_mosaic2,img1,min_x,min_y,f_homo13);
img_mosaic2 = mosaic_func(img_mosaic2,img2,min_x,min_y,f_homo23);
img_mosaic2 = mosaic_func(img_mosaic2,img4,min_x,min_y,f_homo43);
img_mosaic2 = mosaic_func(img_mosaic2,img5,min_x,min_y,f_homo53);
img_mosaic2(2-min_x:row1+1-min_x, 2-min_y:col1+1-min_y,:) =img3;

figure(7)
imtool(uint8(img_mosaic2));
imwrite(img_mosaic2,'E:\2016Fall\661\HW5\mosaic.jpg');