function [output,corner_location] = non_max_suppress(corner_response,corner_response2)
%Non-max suppression method
res_size = size(corner_response);
row = res_size(1);
col = res_size(2);
output = zeros(row,col);
count = 0;
thresh = mean(mean(abs(corner_response)));
win_size = 10; 
for i = win_size + 1:1:row - win_size - 1
    for j = win_size + 1:1:col - win_size - 1
        output(i,j) = (corner_response(i,j) == max(max(corner_response(i-win_size:1:i+win_size,j-win_size:1:j+win_size))) && corner_response(i,j) > thresh);
        if (output(i,j) == 1)
            count = count + 1;
            corner_location{1,count} = [i,j];
        end
    end
end