function result = TextSeg(img)
N = [3,5,7]; %three different window sizes
img_gray = rgb2gray(img);
picSize = size(img_gray);

for count  = 1:1:max(size(N))
    temp_result = zeros(picSize(1),picSize(2));
    win_half = (N(count) - 1)/2;
    for i = 1 + win_half:picSize(1)-win_half
        for j = 1 + win_half:picSize(2)-win_half
            zone = img_gray(i - win_half:i + win_half, j - win_half:j + win_half);
            u = mean(mean(zone));
            temp_result(i,j) = sum(sum(zone - u)).^2 / numel(zone);
        end
    end
    result(:,:,count) = temp_result;
    result(:,:,count) = round(result(:,:,count)*255/max(max(result(:,:,count))));%map it to the range of 255

end

end