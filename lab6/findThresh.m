function final_img = findThresh(img,iter)
%iput as image and number of iterations
L = 256;
picSize = size(img);
%N = picSize(1)*picSize(2);
map = ones(picSize(1),picSize(2)); %the map use to record each iterations result

for count = 1:1:iter

        N = nnz(map); %update the number of pixels involved in each iteration!!!!
        pi = zeros(1,L);%each col represent a value, from 0 to 255
        for i = 1:1:picSize(1) %row
            for j = 1:1:picSize(2) %col
                if (map(i,j) == 1) %accept for calculate
                  pi(1,img(i,j) + 1) = pi(1,img(i,j) + 1) + 1;
                end
            end
        end

        pi = pi/N; %probability distribution of each pixel

        w0 = zeros(1,L); %occurance of two classes, with different threshold k
        w1 = zeros(1,L);
        for k = 1:1:L %threshold 
            w0(1,k) = sum(pi(1,1:k));
            w1(1,k) = 1 - w0(1,k);
        end

        u0 = zeros(1,L);%calcute the mean matrix with different threshold for each class
        u1 = zeros(1,L);
        uT = 0;
        for k = 1:1:L
            uT = uT + k * pi(1,k); %compute the total mean level
        end

        for k = 1:1:L
            uk = 0;
            for temp = 1:1:k
                uk = uk + temp * pi(1,temp);
            end

                u0(1,k) = uk/w0(1,k);
            %end
            u1(1,k) = (uT - uk)/w1(1,k);
        end
 
        %compute the between class varience with respect to the threshold k
        VarBetween = w0 .* w1 .*(u1 - u0).^2;
        threshold = find(VarBetween == max(VarBetween));
        result = zeros(picSize(1),picSize(2));
        for i = 1:1:picSize(1)
             for j = 1:1:picSize(2)
                if (map(i,j) == 1 && img(i,j) > threshold(1));%notice that threshold could be multiple values, pick the 1st one, which is the smallest
                    result(i,j) = img(i,j);
                else
                    map(i,j) = 0;%disable it for attending the next round of calculation
                end
             end
        end
end
    final_img = result;
end