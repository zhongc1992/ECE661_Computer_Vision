clear; 
clc;

flag = 2;
Data1_dir = 'E:\2016Fall\661\HW9\Dataset1\';
Data2_dir = 'E:\2016Fall\661\HW9\Dataset2\';
if (flag == 1)
    Data_dir = Data1_dir;
else
    Data_dir = Data2_dir;
end

Files = dir(strcat(Data_dir,'*.jpg'));%extract all the files
LengthFiles = length(Files);%count total file numbers
fName = {Files(:).name};%extract each file's name

%Initialize parameters
[Allhomo,Allcorner,w_mat,loc_world] = initialize_func(Data_dir);

%Solve for the intrinsic matrix
K = IntrinsicSolve(w_mat);

%Generate input information for LM
p = zeros(1,5+6*LengthFiles);%first 5 parameters from K, and 6 other parameters from each image
p(1:5) = [K(1,1) K(1,2) K(1,3) K(2,2) K(2,3)];

%Solve for extrinsic parameters
[p,img_data,t_cell,R_cell] = ExtrinsicSolve(LengthFiles,Allhomo,Allcorner,p,K);

temp = loc_world';
world_data = temp(:)';%the real world coordinate

%Record parameters after refine
R_refine = cell(1,LengthFiles);
t_refine = cell(1,LengthFiles);

%Calculate for LM
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
p1 = lsqnonlin(@calc_func,p,[],[],options,world_data,img_data,LengthFiles);
alpha_x = p1(1);
s = p1(2);
x0 = p1(3);
alpha_y = p1(4);
y0 = p1(5);
K1 = [alpha_x s x0; 0 alpha_y y0; 0 0 1];
count = 5;

%Recompute the refined extrinsic parameters
for k = 1:LengthFiles
     w = p1(count+1:count+3);
     t_refine{k} = p1(count+4:count+6)';
     count = count + 6;
     wx = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
     phi = norm(w);
     R_refine{k} = eye(3)+(sin(phi)/phi)*wx + ((1-cos(phi))/phi)*wx^2;
end

%%%%%%%%%%%Test and plot
%Select the first image in each dataset as the fixed image, choose other
%images to project onto the fixed image

img_dir = [Data_dir fName{1}];%Pick the first image in dataset as the fixed image
source = 8;%use which image's corner to reproject onto fixed image
[mean1,var1] = MapPlot(K,img_dir,R_cell{1},t_cell{1},Allcorner{1},R_cell{source},t_cell{source},Allcorner{source});
[mean2,var2] = MapPlot(K1,img_dir,R_refine{1},t_refine{1},Allcorner{1},R_refine{source},t_refine{source},Allcorner{source});
