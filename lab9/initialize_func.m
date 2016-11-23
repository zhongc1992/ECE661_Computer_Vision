function [Allhomo,Allcorner,w_mat,loc_world] = initialize_func(Data_dir)
%The function detect image corners' coordinates and solve for absolute
%conic parameters

Files = dir(strcat(Data_dir,'*.jpg'));%extract all the files
LengthFiles = length(Files);%count total file numbers
fName = {Files(:).name};%extract each file's name

Allhomo = cell(1,LengthFiles); % Cell to store homographies
Allcorner = cell(1,LengthFiles); % Cell to store the coordinates of corners

for Number = 1:1:LengthFiles;
     img_dir = [Data_dir fName{Number}];
     im_RGB = imread(img_dir);
     im_gray = rgb2gray(im_RGB); 
     [Height,Width] = size(im_gray); % Size of image
     
     %Canny edge detection 
     BW = edge(im_gray,'canny',0.7); % Binary image from Canny detector
     [H,T,R] = hough(BW); % Hough transform
     P = houghpeaks(H,18,'Threshold',1);
     lines = houghlines(BW,T,R,P,'FillGap',60,'MinLength',60);
     linepara = zeros(length(lines),2);
     for ind = 1:length(lines)
         displacement = lines(ind).point2 - lines(ind).point1;
         a = displacement(2)/displacement(1); % Slope
         if a == inf
            b = inf;
         else
            b = lines(ind).point1(2) - a*lines(ind).point1(1);
         end
         linepara(ind,1) = a; 
         linepara(ind,2) = b; 
     end
     
     %Corner detection steps
     %Start by finding the horizontal lines
     [S,Index] = sort(abs(linepara(:,1)));
     a_Horz = linepara(Index(1:10),1); 
     b_Horz = linepara(Index(1:10),2); 
     [b_Horz,Index] = sort(b_Horz);
     a_Horz = a_Horz(Index);
     
     %find corners and its corresponding labels
     loc_cor = corner(im_gray,80);%totally 80 corners
     label = zeros(80,1);
     
     for ind2 = 1:10
         temp = (a_Horz(ind2)*loc_cor(:,1)-loc_cor(:,2)+b_Horz(ind2)).^2/(a_Horz(ind2)^2+1);
         [temp,Index] = sort(temp);
         Index = Index(1:8);
         [S,J] = sort(loc_cor(Index,1));
         label(8*(ind2-1)+1:8*ind2) = Index(J);
     end
     
     Allcorner{Number} = loc_cor(label,:);

    %%% Solve for homography
     for ind3 = 1:10
         %distance based on 20mm measured in real world
         loc_world(8*(ind3-1)+1:8*ind3,1) = 0:20:140;
         loc_world(8*(ind3-1)+1:8*ind3,2) = 20*(ind3-1);
     end
     A = zeros(2*80,9);
     
     for ind4 = 1:80
         A(2*ind4-1:2*ind4,:)=...
         [0,0,0,-[loc_world(ind4,:),1],loc_cor(label(ind4),2)*[loc_world(ind4,:),1];
         [loc_world(ind4,:),1],0,0,0,-loc_cor(label(ind4),1)*[loc_world(ind4,:),1]];
     end
     [U,D,V] = svd(A);
     h = V(:,9); % Homography estimated by LLS
     Allhomo{Number} = vectomat(h,3); % Save   homographies, vectomat is the same as vec2mat

     v12 = [h(1)*h(2),h(1)*h(5)+h(4)*h(2),h(4)*h(5),h(7)*h(2)+h(1)*h(8),h(7)*h(5)+h(4)*h(8),h(7)*h(8)];
     v11 = [h(1)*h(1),h(1)*h(4)+h(4)*h(1),h(4)*h(4),h(7)*h(1)+h(1)*h(7),h(7)*h(4)+h(4)*h(7),h(7)*h(7)];
     v22 = [h(2)*h(2),h(2)*h(5)+h(5)*h(2),h(5)*h(5),h(8)*h(2)+h(2)*h(8),h(8)*h(5)+h(5)*h(8),h(8)*h(8)];
     BigMat(2*Number-1:2*Number,:) = [v12;v11-v22];
end
[U,D,V] = svd(BigMat); %solve for null vector by using SVD
w_mat = V(:,6);