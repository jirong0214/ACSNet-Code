function [initResidual] = MutiScaleResidual(image)

%compute Entropy of the block;
ImEntropy09 = ComputeEntropy(image);
ImEntropy = roundn(ImEntropy09,-3);
global Strategy2EArray;
index = find(Strategy2EArray == ImEntropy);
rank =(index)/length(Strategy2EArray);
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

% fprintf("Index of this block is %g\n",index);
% format long g;
% fprintf("Entropy of this block is %g\n",ImEntropy);
% format long g;
% fprintf('SampleRate of this block is %g\n',Rate);
% format long g;
save('SampleRate.txt','Rate','-append','-ascii');
end

