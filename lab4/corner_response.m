function output = corner_response(C,corner_map);
%Generate the corner response
k = 0.04;% defined constant k
img_size = size(C);
row = img_size(1);
col = img_size(2);
output = zeros(row,col);

for i = 1:1:row
    for j = 1:1:col
        if (corner_map(i,j) == 1)
            output(i,j) = det(C{i,j}) - k*(trace(C{i,j}))^2;
        end
    end
end