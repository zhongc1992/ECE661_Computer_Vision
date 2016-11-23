function ERROR = calc_func(p,World_coord,Image_coord,number_img)
    %Getting parameters from vector p
    alpha_x = p(1);
    s = p(2);
    x0 = p(3);
    alpha_y = p(4);
    y0 = p(5);
    %Build intrinsic matrix
    Image_estimate = zeros(1,number_img*160);
    K = [alpha_x s x0; 0 alpha_y y0; 0 0 1];
    count = 5;
    index1=1;
    
    for ind = 1:number_img
         w_vector = p(count+1:count+3);%extract parameters 
         t_vector = p(count+4:count+6)';
         count = count + 6;
         wx = [0 -w_vector(3) w_vector(2); w_vector(3) 0 -w_vector(1); -w_vector(2) w_vector(1) 0];
         phi = norm(w_vector);
         R = eye(3)+sin(phi)/phi*wx + (1-cos(phi))/phi*wx^2;%compute rotation matrix
         index2=1;
         for i = 1:80
             x = K*[R t_vector]*[World_coord(index2:index2+1) 0 1]';
             Image_estimate(index1:index1+1) = [x(1)/x(3) x(2)/x(3)];
             index1 = index1+2;
             index2 = index2+2;
         end
    end
    ERROR = Image_coord - Image_estimate;%error between real and estimated result