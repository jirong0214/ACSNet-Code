function initRec = ACS_Net_initRec(image,baseSR)
numSubNet = 2;  %0.05 Sample rate
netfolder = '../BigData/model/SubNetInit';
netpaths = dir(fullfile(netfolder,['subnetInit0',num2str(numSubNet),'.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

global showDetailMsg;
global useGPU;
if useGPU
net.move('gpu');
end

Rate = baseSR;

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
    s01Init         = gather(net.vars(net.getVarIndex('s01combine')).value);
    s02TotalInit = gather(net.vars(net.getVarIndex('s02combineAdds01')).value);
else
    s01Init         = net.vars(net.getVarIndex('s01combine')).value;
    s02TotalInit = net.vars(net.getVarIndex('s02combineAdds01')).value;
end

switch Rate
    case 0.01
        initRec =  s01Init;
    case 0.05
        initRec = s02TotalInit; 
end
if showDetailMsg == 1
    fprintf('Basic Sample Rate is %d\n',Rate);
end
end