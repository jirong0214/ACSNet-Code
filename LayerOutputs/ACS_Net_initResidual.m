function initResidual = ACS_Net_initResidual(image,baseSR)
tic
%%选择训练的模型路径：
netfolder = '../Bigdata/TrainedModel';% tianjirong's trained model folder；
netpaths = dir(fullfile(netfolder,['net-epoch-50.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name));
net = dagnn.DagNN.loadobj(net.net);
%选择是否使用GPU！
useGPU      = 0;
if useGPU 
    net.move('gpu');
end
ImEntropy09 = ComputeEntropy(image);%计算当前块的熵；
ImEntropy = roundn(ImEntropy09,-3);%保留3位小数；
global Strategy2EArray;%全局变量：获取主函数里的熵数组；
index = find(Strategy2EArray == ImEntropy);
fprintf("index为：%d\n",index);
fprintf("小块的熵为：%d\n",ImEntropy);
%%根据图像的熵排位来switch采样率；
rank =(index+1)/length(Strategy2EArray);%计算该块的熵排位
if rank < 0.4
    Rate = 0.05;
elseif  rank <0.8
    Rate = 0.1;
elseif rank < 0.9 
    Rate = 0.2;
else
    Rate = 0.2;
end

%图像转为单精度灰度图；
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
    %%使用GPU的情况下和CPU同理:
    %%各层的初始重构输出：↓
    s01initOutput = gather(net.vars(net.getVarIndex('s01combine'))).value;%  第1层的初始重构图像；
    s02initOutput = gather(net.vars(net.getVarIndex('s02combine'))).value;% 第2层的初始重构残差；
    s03initOutput = gather(net.vars(net.getVarIndex('s03combine'))).value;%第3层的初始重构残差;
    s04initOutput = gather(net.vars(net.getVarIndex('s04combine'))).value;%第4层的初始重构残差
    s05initOutput = gather(net.vars(net.getVarIndex('s05combine'))).value;%第5层的初始重构残差
    s06initOutput = gather(net.vars(net.getVarIndex('s06combine'))).value;%第6层的初始重构残差
    s07initOutput =  gather(net.vars(net.getVarIndex('s07combine'))).value;%第7层的初始重构残差
else
    %%各层的初始重构输出：↓
    s01initOutput = net.vars(net.getVarIndex('s01combine')).value;%  第1层的初始重构图像；
    s02initOutput = net.vars(net.getVarIndex('s02combine')).value;% 第2层的初始重构残差；
    s03initOutput = net.vars(net.getVarIndex('s03combine')).value;%第3层的初始重构残差;
    s04initOutput = net.vars(net.getVarIndex('s04combine')).value;%第4层的初始重构残差
    s05initOutput = net.vars(net.getVarIndex('s05combine')).value;%第5层的初始重构残差
    s06initOutput = net.vars(net.getVarIndex('s06combine')).value;%第6层的初始重构残差
    s07initOutput = net.vars(net.getVarIndex('s07combine')).value;%第7层的初始重构残差
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
            initResidual  = zeros(size(image));
            
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
fprintf('当前块的采样率为：%g\n',Rate);
format long g;
fprintf('当前块的熵为：%g\n',ImEntropy);
format long g;
save('SampleRate.txt','Rate','-append','-ascii');
toc
end

