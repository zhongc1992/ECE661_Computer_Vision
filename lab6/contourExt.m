function output = contourExt(input,win_size)
%input - input image
%win_size - window size
picSize = size(input);
output = zeros(picSize(1),picSize(2));
win_half = (win_size - 1)/2;

for i = 1 + win_half:picSize(1) - win_half
    for j = 1 + win_half:picSize(2) - win_half
        if input(i,j) == 0
        else 
            zone = input(i - win_half:i + win_half, j - win_half:j + win_half);
            if (input(i,j) == 1 && sum(zone(:) == 0) > 0);%if pixel value is 1 and one or more of its adjacent has value more than 0
                output(i,j) = 1;%it is the contour
            end
        end
    end
end