function runs = runCount(input)
%Count the runs() of a binary pattern 
length = max(size(input));
flag = input(1);
runs = 1;
for i = 1:1:length
    if (flag == input(i))
    else
        runs = runs + 1;
        flag = input(i);
    end
end

end