function [residual03] = getLayerResidual03(image)
% Get Third Layer Residual From SubNet;
    netfolder = '../BigData/model/SubNetInit';
    netpaths = dir(fullfile(netfolder,['subNetInit03.mat'])); 
    net = load(fullfile(netfolder,netpaths.name)); 
    net = dagnn.DagNN.loadobj(net.net);
    global useGPU;
if useGPU 
    net.move('gpu');
end
if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2single(image(:, :, 1));
else
    image =im2single(image);
end
image = modcrop(image,32);
if useGPU
    input = gpuArray(image);
else
    input = image;
end
net.conserveMemory = false;
net.eval({'input',input});

if useGPU
    residual03 = gather(net.vars(net.getVarIndex('s03combine')).value);
else
    residual03 = net.vars(net.getVarIndex('s03combine')).value;
end

end

