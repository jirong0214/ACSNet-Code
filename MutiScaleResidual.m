function [initResidual] = MutiScaleResidual(image,entropyVector,sortedEntropyVector)
global blockNo;
blockNo = blockNo + 1;
global showDetailMsg;
%find the index and distribute Sample Rate by it's rank;
index = find(sortedEntropyVector == entropyVector(blockNo));
rank =(index)/length(sortedEntropyVector);

if rank < 0.45
    Rate = 0.05;
elseif  rank <0.85
    Rate = 0.1;
elseif rank < 0.95
    Rate = 0.2;
else
    Rate = 0.3;
end

%Set Rate == 0.1 to compare results
%Rate = 0.1;

%distribute Sample rate and return Residual;
if Rate == 0.05
    initResidual = zeros(size(image));
elseif Rate == 0.1
    initResidual = getLayerResidual03(image);
elseif Rate == 0.2
    initResidual = getLayerResidual04(image);
elseif Rate == 0.3
    initResidual = getLayerResidual05(image);
end

if showDetailMsg == 1
    disp(num2str(blockNo));
    fprintf("Index of this block is %g\t",index);
    format long g;
    fprintf('SampleRate of this block is %g\n',Rate);
    format long g;
end
save('SampleRate.txt','Rate','-append','-ascii');
end

