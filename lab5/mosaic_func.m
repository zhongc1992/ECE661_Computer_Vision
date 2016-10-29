function result = mosaic_func(mosaic,img,off_row,off_col,homo)
    mosaic = double(mosaic);
    mosaic_size = size(mosaic);
    mosaic_row = mosaic_size(1);
    mosaic_col = mosaic_size(2);
    
    img_size = size(img);
    row = img_size(1);
    col = img_size(2);
    
    for i = 1:1:mosaic_row
        for j = 1:1:mosaic_col
              %pretend to be img3's coordinate
              fake_x = i + off_row;
              fake_y = j + off_col;
              %look from mosaic image to img to find pixel value
              pt_hat = homo\[fake_x,fake_y,1]';
              pt_2d(1) = pt_hat(1)/pt_hat(3);
              pt_2d(2) = pt_hat(2)/pt_hat(3);
              
              %take round to make coordinate to be integers, performance
              %the roughly same as bilinear interpolation
              pt_2d(1) = round(pt_2d(1));
              pt_2d(2) = round(pt_2d(2));
              if (pt_2d(1)<=row && pt_2d(1)>=1 && pt_2d(2)<=col && pt_2d(2)>=1)
                mosaic(i,j,:) = img(pt_2d(1),pt_2d(2),:);
              end

        end
    end
    result = uint8(mosaic);

end