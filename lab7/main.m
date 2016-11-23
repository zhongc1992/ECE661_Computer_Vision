iter = 10;
%Read images from .txt files
img1 = dlmread('depthImage1ForHW.txt');
img2 = dlmread('depthImage2ForHW.txt');
picSize = size(img1);
row = picSize(1);
col = picSize(2);

K = [365,0,256;0,365,212;0,0,1];%intrinsic camera matrix

cloud_1 = cell(row,col);
cloud_2 = cell(row,col);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate point clound
%point u = (x,y,1) is defind as [row;col;1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:1:row
    for j = 1:1:col
        u = [i;j;1];
        cloud_1{i,j} = img1(i,j) * inv(K) * u;
        cloud_2{i,j} = img2(i,j) * inv(K) * u;
    end
end
save('P_original.mat','cloud_1');
save('Q_original.mat','cloud_2');
for i = 1:1:iter
    name = ['Q_' num2str(i)];
   if i == 1
       Q_refined = ICP(cloud_1,cloud_2,i);
       save(name,'Q_refined');
   else
       Q_refined = ICP(cloud_1,Q_refined,i);
       save(name,'Q_refined');
   end
end