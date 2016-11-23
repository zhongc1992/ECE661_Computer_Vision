fileBuilding = 'E:\2016Fall\661\HW8\imagesDatabaseHW8\training\building';
fileCar = 'E:\2016Fall\661\HW8\imagesDatabaseHW8\training\car';
fileMountain = 'E:\2016Fall\661\HW8\imagesDatabaseHW8\training\mountain';
fileTree = 'E:\2016Fall\661\HW8\imagesDatabaseHW8\training\tree';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_mat = [];
for i = 1:1:4
    if i == 1
        folder = fileBuilding;
    elseif i == 2
        folder = fileCar;
    elseif i == 3
        folder = fileMountain;
    elseif i == 4
        folder = fileTree;
    end
    
    Files = dir(strcat(folder,'\*.jpg'));
    LengthFiles = length(Files);
    fName = {Files(:).name};
    for j = 1:LengthFiles
        address = [folder '\' fName{1,j}];
        hist_result = LBP_func(address);
        training_mat = [training_mat; hist_result];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 5;%define k nearest neighbor
fileTest = 'E:\2016Fall\661\HW8\imagesDatabaseHW8\testing';
Files = dir(strcat(fileTest,'\*.jpg'));
LengthFiles = length(Files);
fName = {Files(:).name};
testResult = [];
for h = 1:1:LengthFiles
    address = [fileTest '\' fName{1,h}];
    hist_result = LBP_func(address);
    eucMat = [];
    for q = 1:1:max(size(training_mat))
        temp = training_mat(q,:);
        eucDis = sqrt(sum((hist_result - temp).^2));
        if q >= 1 && q<= 20
            label = 1; %label as building
        elseif q > 20 && q <= 30
            label = 2; %label as car
        elseif q > 40 && q <= 60
            label = 3;
        elseif q > 60 && q <= 80
            label = 4; %label as tree
        end          
        eucMat(q,1) = eucDis;%record euclidean distance
        eucMat(q,2) = label;% record label
    end
    for ind = 1:1:k %find k nearest neighbor
        [loc_r,loc_c]  = find(eucMat(:,1) == min(eucMat(:,1)));
        loc_r = loc_r(1);%only take one as smallest once a time
        loc_c = loc_c(1);
        testResult(h,ind) = eucMat(loc_r,2);%record the label to the result matrix
        eucMat(loc_r,:) = [];%delete it from euc matrix
    end
end