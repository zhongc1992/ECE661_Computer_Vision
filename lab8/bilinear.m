function output = bilinear(row,delta1,col,delta2,image)
%bilinear interpolation for a point(row,col) in the image, delta1 and
%delta2 are the displacements in row,col directions.
    picSize = size(image);
    height = picSize(1);
    width = picSize(2);

    x = row + delta1;
    y = col + delta2;
    Ax = row;
    Ay = col;
    Bx = Ax;
    By = round(Ay + delta2);
    Cx = round(Ax + delta1);
    Cy = Ay;
    Dx = round(Ax + delta1);
    Dy = round(Ay + delta2);
    
    output = image(Ax,Ay) * (1 - abs(Ax - x)) * (1 - abs(Ay - y)) + image(Bx,By) * (1-abs(Bx - x)) * (1- abs(By-y)) + image(Cx,Cy) * (1-abs(Cx-x)) * (1-abs(Cy-y)) + image(Dx,Dy) * (1-abs(Dx-x)) * (1-abs(Dy-y));
end