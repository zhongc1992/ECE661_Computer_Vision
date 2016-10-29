function corner_map = check_rank(C);
%Take matrix C as input, check if its rank equals to two
img_size = size(C);
row = img_size(1);
col = img_size(2);
corner_map = zeros(row,col);

for i = 1:1:row
    for j = 1:1:col
        corner_map(i,j) = (rank(C{i,j}) == 2);
    end
end