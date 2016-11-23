function [p,img_data,t_cell,R_cell] = ExtrinsicSolve(LengthFiles,Allhomo,Allcorner,p,K)
%Compute Extrinsic and LM Parameters for refining
img_data = [];%input for LM
t_cell = cell(1,LengthFiles);% Cell to store translation matrix
R_cell = cell(1,LengthFiles);% Cell to store ratation matrix for matrix
count = 5;%count for the number index for p vector
for k = 1:LengthFiles
     H = Allhomo{k};
     scale_factor = 1/(norm(K\H(:,1)));
     t = K\H(:,3);
     r1 = scale_factor*K\H(:,1);
     r2 = scale_factor*K\H(:,2);
     r3 = cross(r1,r2);
     t = t*scale_factor;
     R = [r1,r2,r3];
     [U,D,V] = svd(R);
     R = U*V';
     t_cell{k} = t;
     R_cell{k} = R;
     
 %Levenberg-Marquardt method
    phi = acos((trace(R)-1)/2);
    w = (phi/(2*sin(phi)))*([R(3,2)-R(2,3) R(1,3)-R(3,1) R(2,1)- R(1,2)])';
    p(count+1:count+3) = w;
    p(count+4:count+6) = t;
    count = count + 6;
    temp = Allcorner{k};
    temp = temp';
    img_data = [img_data temp(:)'];%the image corners' coordinates
end
