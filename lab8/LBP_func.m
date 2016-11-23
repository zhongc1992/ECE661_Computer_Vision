function hist_mat = LBP_func(address)
%Find the LBP feature for a given image
img = imread(address);
img_gray = double(rgb2gray(img));

picSize = size(img_gray);
row = picSize(1);
col = picSize(2);

P = 8;%Define as 8 neighbor
R = 1;%Define as radius equals to 1

p = 0:7;
delta_u = R * cos(2*pi*p./P);%vertical difference, pointing down
delta_v = R * sin(2*pi*p./P);%horizontal difference, pointing to right

hist_mat = zeros(1,P+2);%initial the histogram vector
for i = 2:1:row - 1
    for j = 2:1:col - 1
        p(1) = img_gray(i+round(delta_u(1)),j+round(delta_v(1)));
        p(2) = bilinear(i,delta_u(2),j,delta_v(2),img_gray);
        p(3) = img_gray(i+round(delta_u(3)),j+round(delta_v(3)));
        p(4) = bilinear(i,delta_u(4),j,delta_v(4),img_gray);
        p(5) = img_gray(i+round(delta_u(5)),j+round(delta_v(5)));
        p(6) = bilinear(i,delta_u(6),j,delta_v(6),img_gray);
        p(7) = img_gray(i+round(delta_u(7)),j+round(delta_v(7)));
        p(8) = bilinear(i,delta_u(8),j,delta_v(8),img_gray);
        center = img_gray(i,j);
        judge = (p>=center); %build the binary format
        value = bitvector(judge);%get the binary sequence which has min integer value
        runs = runCount(value);
        if runs > 2
            encode = P + 1;
        elseif runs == 2
            encode = sum(nonzeros(value));
        elseif (runs == 1) && (value(1) == 0)
            encode = 0;
        elseif (runs == 1) && (value(1) == 1)
            encode = P;
        end
        hist_mat(encode + 1) = hist_mat(encode + 1) + 1;       
    end
end
