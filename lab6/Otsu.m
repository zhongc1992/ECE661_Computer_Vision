clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Image segmentation for image 1 -- the lake image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img1 = imread('E:\2016Fall\661\HW6\lake.jpg');
img1red = img1(:,:,1);
img1green = img1(:,:,2);
img1blue = img1(:,:,3);
input1red = double(img1red);
input1green = double(img1green);
input1blue = double(img1blue);

%Define size of image 1
pic1size = size(img1);
N1 = pic1size(1)*pic1size(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image 1 RGB based result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seg1_red =  findThresh(input1red,1);
seg1_green = findThresh(input1green,1);
seg1_blue = findThresh(input1blue,1);
% figure(1)
% imshow(seg1_red);
% title('Image segmentation on image 1 Red channel');
% figure(2)
% imshow(seg1_green);
% title('Image segmentation on image 1 Green channel');
% figure(3)
% imshow(seg1_blue);
% title('Image segmentation on image 1 Blue channel');

%Construct the final result on image1 for RGB channel
img1_RGB = zeros(pic1size(1),pic1size(2));
for i = 1:1:pic1size(1)
    for j = 1:1:pic1size(2)
        if (~seg1_red(i,j) > 0 && ~seg1_green(i,j) > 0 && seg1_blue(i,j) > 0)
            img1_RGB(i,j) = 1;
        end
    end
end
% figure(4)
% imshow(img1_RGB);
% title('Final result over RGB channel for image1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image 1 texture based result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% texture = TextSeg(img1);%texture(:,:,1) is N = 3.texture(:,:,2) is N = 5, texture(:,:,3) is N =7
% seg1_N3 =  findThresh(texture(:,:,1),1);
% seg1_N5 = findThresh(texture(:,:,2),1);
% seg1_N7 = findThresh(texture(:,:,3),1);
% 
% figure(50)
% imshow(seg1_N3);
% title('Image segmentation on image 1 with N = 3');
% figure(51)
% imshow(seg1_N5);
% title('Image segmentation on image 1 with N = 5');
% figure(52)
% imshow(seg1_N7);
% title('Image segmentation on image 1 with N = 7');

% img1_comb = zeros(pic1size(1),pic1size(2));
% for i = 1:1:pic1size(1)
%     for j = 1:1:pic1size(2)
%         if (~seg1_N3(i,j) > 0 && ~seg1_N5(i,j) > 0 && ~seg1_N7(i,j) > 0)
%             img1_comb(i,j) = 1;
%         end
%     end
% end
% 
% figure(53)
% imshow(img1_comb);
% title('Final result using texture for image1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Image 1 ends here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Image segmentation for image 2 starts here -- leopard image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img2 =  imread('E:\2016Fall\661\HW6\leopard.jpg');
pic2size = size(img2);
texture = TextSeg(img2);%texture(:,:,1) is N = 3.texture(:,:,2) is N = 5, texture(:,:,3) is N =7
seg2_N3 =  findThresh(texture(:,:,1),1);
seg2_N5 = findThresh(texture(:,:,2),1);
seg2_N7 = findThresh(texture(:,:,3),1);
% figure(5)
% imshow(seg2_N3);
% title('Image segmentation on image 2 with N = 3');
% figure(6)
% imshow(seg2_N5);
% title('Image segmentation on image 2 with N = 5');
% figure(7)
% imshow(seg2_N7);
% title('Image segmentation on image 2 with N = 7');

%Construct the final result on image1 for RGB channel
img2_comb = zeros(pic2size(1),pic2size(2));
for i = 1:1:pic2size(1)
    for j = 1:1:pic2size(2)
        if (seg2_N3(i,j) > 0 && seg2_N5(i,j) > 0 && seg2_N7(i,j) > 0)
            img2_comb(i,j) = 1;
        end
    end
end
% figure(8)
% imshow(img2_comb);
% title('Final result overall channel for image2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Image segmentation for image 2 use RGB method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img2red = img2(:,:,1);
% img2green = img2(:,:,2);
% img2blue = img2(:,:,3);
% input2red = double(img2red);
% input2green = double(img2green);
% input2blue = double(img2blue);
% seg2_red =  findThresh(input2red,1);
% seg2_green = findThresh(input2green,1);
% seg2_blue = findThresh(input2blue,1);
% img2_RGB = zeros(pic2size(1),pic2size(2));
% for i = 1:1:pic2size(1)
%     for j = 1:1:pic2size(2)
%         if (seg2_red(i,j) > 0 && seg2_green(i,j) > 0 && seg2_blue(i,j) > 0)
%             img2_RGB(i,j) = 1;
%         end
%     end
% end
% figure(51)
% imshow(seg2_red);
% title('Image segmentation on image 2 Red channel');
% figure(52)
% imshow(seg2_green);
% title('Image segmentation on image 2 Green channel');
% figure(53)
% imshow(seg2_blue);
% title('Image segmentation on image 2 Blue channel');
% figure(54)
% imshow(img2_RGB);
% title('Final result over RGB channel for image2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image2 segmentation ends here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Image segmentation for image 3 starts here -- brain image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img3 =  imread('E:\2016Fall\661\HW6\brain.jpg');
pic3size = size(img3);
img3_gray = double(rgb2gray(img3));

seg3 =  findThresh(img3_gray,3);
% figure(19)
% imshow(img3_gray);
img3_comb = (seg3 > 0);
% figure(19)
% imshow(img3_comb);
% title('The segmented result of brain white matter after 3 iteration of otsu method');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image3 segmentation texture method 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
texture = TextSeg(img3);%texture(:,:,1) is N = 3.texture(:,:,2) is N = 5, texture(:,:,3) is N =7
seg3_N3 =  findThresh(texture(:,:,1),1);
seg3_N5 = findThresh(texture(:,:,2),1);
seg3_N7 = findThresh(texture(:,:,3),1);
img3_text = zeros(pic3size(1),pic3size(2));
for i = 1:1:pic3size(1)
    for j = 1:1:pic3size(2)
        if (seg3_N3(i,j) > 0 && seg3_N5(i,j) > 0 && seg3_N7(i,j) > 0)
            img3_text(i,j) = 1;
        end
    end
end

figure(51)
imshow(seg3_N3);
title('Image segmentation on image 3 N =3 channel');
figure(52)
imshow(seg3_N5);
title('Image segmentation on image 3 N =5 channel');
figure(53)
imshow(seg3_N7);
title('Image segmentation on image 3 N =7 channel');
figure(54)
imshow(img3_text);
title('Texture method for image 3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image3 segmentation ends here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the contour of the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Remove noise from background
se = strel('square',10);
eros1 = imerode(img1_RGB,se);
noise_B1 = imdilate(eros1,se);



se = strel('square',3);
eros3 = imerode(img3_comb,se);
noise_B3 = imdilate(eros3,se);



% figure(9)
% imshow(noise_B);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%remove noise from foreground
se = strel('square',5);
dila1 = imdilate(noise_B1,se);
noiseFree1 = imerode(dila1,se);

se = strel('square',10);
dila2 = imdilate(img2_comb,se);
noiseFree2 = imerode(dila2,se);
% dila2 = imdilate(img2_RGB,se);
% noiseFree2 = imerode(dila2,se);

se = strel('square',15);
dila3 = imdilate(noise_B3,se);
noiseFree3 = imerode(dila3,se);
% figure(10)
% imshow(noiseFree);

%Apply the final mask on original image
for i = 1:1:3
    img1(:,:,i) = double(img1(:,:,i)) .* noiseFree1;
    img2(:,:,i) = double(img2(:,:,i)) .* noiseFree2;
    img3(:,:,i) = double(img3(:,:,i)) .* noiseFree3;
end


figure(13)
imshow(noiseFree1);
title('the noise free mask for image 1');
figure(14)
imshow(img1);
title('Corresponding foreground on image 1');

figure(16)
imshow(noiseFree2);
title('the noise free mask for image 2');
figure(17)
imshow(img2);
title('Corresponding foreground on image 2');

figure(20)
imshow(noiseFree3);
title('the noise free mask for image 3');
figure(21)
imshow(img3);
title('Corresponding foreground on image 3');
%find the contour 
contour1 = contourExt(noiseFree1,3);
contour2 = contourExt(noiseFree2,3);
contour3 = contourExt(noiseFree3,3);

figure(15)
imshow(contour1);
title('Corresponding contour on image 1');
figure(18)
imshow(contour2);
title('Corresponding contour on image 2');
figure(22)
imshow(contour3);
title('Corresponding contour on image 3');