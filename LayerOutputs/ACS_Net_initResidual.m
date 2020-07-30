function initResidual = ACS_Net_initResidual(image)
netfolder = '../Bigdata/model/TrainedModel';% tianjirong's trained model folder
netpaths = dir(fullfile(netfolder,['net-epoch-50.mat'])); %switch net path
net = load(fullfile(netfolder,netpaths.name));
net = dagnn.DagNN.loadobj(net.net);
baseSR =0.05;
global useGPU;
if useGPU 
    net.move('gpu');
end

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
    s01initOutput = gather(net.vars(net.getVarIndex('s01combine')).value);
    s02initOutput = gather(net.vars(net.getVarIndex('s02combine')).value);
    s03initOutput = gather(net.vars(net.getVarIndex('s03combine')).value);
    s04initOutput = gather(net.vars(net.getVarIndex('s04combine')).value);
    s05initOutput = gather(net.vars(net.getVarIndex('s05combine')).value);
    s06initOutput = gather(net.vars(net.getVarIndex('s06combine')).value);
    s07initOutput = gather(net.vars(net.getVarIndex('s07combine')).value);
else
    s01initOutput = net.vars(net.getVarIndex('s01combine')).value;
    s02initOutput = net.vars(net.getVarIndex('s02combine')).value;
    s03initOutput = net.vars(net.getVarIndex('s03combine')).value;
    s04initOutput = net.vars(net.getVarIndex('s04combine')).value;
    s05initOutput = net.vars(net.getVarIndex('s05combine')).value;
    s06initOutput = net.vars(net.getVarIndex('s06combine')).value;
    s07initOutput = net.vars(net.getVarIndex('s07combine')).value;
end
if baseSR == 0.01
    switch Rate
        case 0.01
            initResidual  = zeros(size(image));
        case 0.05
            initResidual = s02initOutput;
        case 0.1
            initResidual = s02initOutput+s03initOutput;
        case 0.2
            initResidual = s02initOutput+s03initOutput+s04initOutput;
        case 0.3
            initResidual = s02initOutput+s03initOutput+s04initOutput+s05initOutput;
        case 0.4
            initResidual = s02initOutput+s03initOutput+s04initOutput+s05initOutput+s06initOutput;
        case 0.5
            initResidual = s02initOutput+s03initOutput+s04initOutput+s05initOutput+s06initOutput+s07initOutput;
    end
end
if baseSR == 0.05
    switch Rate
        case 0.01
            initResidual = zeros(size(image));
        case 0.05
            initResidual = zeros(size(image));
        case 0.1
            initResidual = s03initOutput;
        case 0.2
            initResidual = s03initOutput+s04initOutput;
        case 0.3
            initResidual = s03initOutput+s04initOutput+s05initOutput;
        case 0.4
            initResidual = s03initOutput+s04initOutput+s05initOutput+s06initOutput;
        case 0.5
            initResidual = s03initOutput+s04initOutput+s05initOutput+s06initOutput+s07initOutput;
    end
end

global PrintProcessMessage;
if PrintProcessMessage == 1
    fprintf("Index of this block is %g\n",index);
    format long g;
    fprintf("Entropy of this block is %g\n",ImEntropy);
    format long g;
    fprintf('SampleRate of this block is %g\n',Rate);
    format long g;
end
save('SampleRate.txt','Rate','-append','-ascii');
end

