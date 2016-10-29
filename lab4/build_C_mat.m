function C = build_C_mat(image_x,image_y,scale)
%construct the C matrix from given image gradients in both direction and haar scale factor
img_size = size(image_x);
row = img_size(1); %height of the image
col = img_size(2); %width of the image
range = 5*scale;
C = cell(row,col);

for i = 1:1:row
    for j = 1:1:col
        C_temp = [0,0;0,0];
        for p = - round((range-1)/2) : 1 : round((range-1)/2)
            for q = -round((range-1)/2) : 1 : round((range-1)/2)
                if ((i + p > 0) && (i + p < row) && (j + q > 0) && (j + q < col))%set up the border case to 0
                    C_temp(1,1) = C_temp(1,1) + image_x(i+p,j+q)^2;
                    C_temp(1,2) = C_temp(1,2) + image_x(i+p,j+q) * image_y(i+p,j+q);
                    C_temp(2,1) = C_temp(2,1) + image_x(i+p,j+q) * image_y(i+p,j+q);
                    C_temp(2,2) = C_temp(2,2) + image_y(i+p,j+q)^2;
                end
            end
        end
        C{i,j} = C_temp;
    end
end
