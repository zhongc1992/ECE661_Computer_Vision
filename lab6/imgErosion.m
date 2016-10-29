function output = imgErosion(input,win_size)
%input - input image
%win_size - the erosion window size
picSize = size(input);
output = zeros(picSize(1),picSize(2));
win_half = (win_size - 1)/2;
for i = 1 + win_half:picSize(1) - win_half
    for j = 1 + win_half:picSize(2) - win_half
        zone = input(i - win_half:i + win_half, j - win_half:j + win_half);
        smallest = min(min(zone));
        output(i,j) = smallest;
    end
end