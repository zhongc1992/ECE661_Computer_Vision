function output = bitvector(p)
%Find the minimum integer value from a binary sequence through left shift
    iter = max(size(p));
    value = [];
    for i = 1:1:iter
        binary = '';
        for j = 1:1:iter
            if (i + j - 1) <= iter
                binary = [binary num2str(p(i+j-1))];
            else
                binary = [binary num2str(p(i+j-1-iter))];
            end
        end
        value(i) = bin2dec(binary);
    end
    loc = find(value == min(value));
    if (max(size(loc)) > 1)
        loc = loc(1);
    end
    for i = 1:1:iter
        if loc + i - 1 <= iter
            output(i) = p(loc + i - 1);
        else 
            output(i) = p(loc + i - 1 - iter);
        end
    end
end