function initRec = ACS_Net_initRec(image,baseSR)
netfolder = '../BigData/TrainedModel';% tianjirong's trained model folder
netpaths = dir(fullfile(netfolder,['net-epoch-50.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

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
label = image;
net.conserveMemory = false;
net.eval({'input',input});


if useGPU
    s01Init         = gather(net.vars(net.getVarIndex('s01combine')).value);
    s02TotalInit = gather(net.vars(net.getVarIndex('s02combineAdds01')).value);
    s03TotalInit = gather(net.vars(net.getVarIndex('s03combineAdds02')).value);
    s04TotalInit = gather(net.vars(net.getVarIndex('s04combineAdds03')).value);
    s05TotalInit = gather(net.vars(net.getVarIndex('s05combineAdds04')).value);
    s06TotalInit = gather(net.vars(net.getVarIndex('s06combineAdds05')).value);
    s07TotalInit = gather(net.vars(net.getVarIndex('s07combineAdds06')).value);
else
    s01Init         = net.vars(net.getVarIndex('s01combine')).value;
    s02TotalInit = net.vars(net.getVarIndex('s02combineAdds01')).value;
    s03TotalInit = net.vars(net.getVarIndex('s03combineAdds02')).value;
    s04TotalInit = net.vars(net.getVarIndex('s04combineAdds03')).value;
    s05TotalInit = net.vars(net.getVarIndex('s05combineAdds04')).value;
    s06TotalInit = net.vars(net.getVarIndex('s06combineAdds05')).value;
    s07TotalInit = net.vars(net.getVarIndex('s07combineAdds06')).value;
end

switch Rate
    case 0.01
        initRec =  s01Init;
    case 0.05
        initRec = s02TotalInit; 
    case 0.1
        initRec = s03TotalInit; 
    case 0.2
        initRec = s04TotalInit; 
    case 0.3
        initRec = s05TotalInit; 
    case 0.4
        initRec = s06TotalInit; 
    case 0.5
        initRec = s07TotalInit; 
end
global PrintProcessMessage;
if PrintProcessMessage == 1
    fprintf('Basic Sample Rate is %d\n',Rate);
end
end