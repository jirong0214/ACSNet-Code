function deepResidual = ACS_Net_deepResidual(image)
tic

%%选择训练的模型路径：
%netfolder = '.\model'; %pretrained model；
%netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained；  
netfolder = 'TrainedModel';% tianjirong's trained model folder；
netpaths = dir(fullfile(netfolder,['net-epoch-50.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

%选择是否使用GPU！
useGPU      = 0;  
if useGPU
net.move('gpu');
end
%选择是否要显示和保存图像重构结果：
showResult  = 0; 
SaveImage = 0;


%%根据图像的熵来switch采样率；
ImEntropy =  entropy(image);  
if ImEntropy < 5
Rate = 0.05;   
elseif  ImEntropy <5.5
Rate = 0.05;   
elseif  ImEntropy <6
Rate = 0.1  ;   
elseif  ImEntropy <6.5
Rate = 0.2  ; 
elseif  ImEntropy <7
Rate = 0.3  ;  
elseif  ImEntropy <7.5
Rate = 0.4  ;  
else
Rate = 0.5  ;   
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


%强制赋值采样率而不是根据图像熵来分配！
%Rate = 0.5; 

if useGPU
    %%使用GPU的情况下和CPU同理，前面加个gather即可，我就先不改了；
switch Rate
    case 0.01
        output = gather(net.vars(net.getVarIndex('s01dr_pred')).value);
    case 0.05
        output = gather(net.vars(net.getVarIndex('s02dr_pred')).value);
    case 0.1
        output = gather(net.vars(net.getVarIndex('s03dr_pred')).value);
    case 0.2
        output = gather(net.vars(net.getVarIndex('s04dr_pred')).value);
    case 0.3
        output = gather(net.vars(net.getVarIndex('s05dr_pred')).value);
    case 0.4
        output = gather(net.vars(net.getVarIndex('s06dr_pred')).value);
    case 0.5
        output = gather(net.vars(net.getVarIndex('s07dr_pred')).value);
end
else
    %%各层的初始重构输出：↓
    s01initOutput = net.vars(net.getVarIndex('s01combine')).value;%  第1层的初始重构图像；
    s02initOutput = net.vars(net.getVarIndex('s02combine')).value;% 第2层的初始重构残差；
    s03initOutput = net.vars(net.getVarIndex('s03combine')).value;%第3层的初始重构残差;
    s04initOutput = net.vars(net.getVarIndex('s04combine')).value;%第4层的初始重构残差
    s05initOutput = net.vars(net.getVarIndex('s05combine')).value;%第5层的初始重构残差
    s06initOutput = net.vars(net.getVarIndex('s06combine')).value;%第6层的初始重构残差
    s07initOutput = net.vars(net.getVarIndex('s07combine')).value;%第7层的初始重构残差
    PSNR_init01 = psnr(s01initOutput,image); 
    SSIM_init01 = ssim(s01initOutput,image);
    
    
    %各层初始残差输出的平均值↓（衡量一下残差的大小,没别的意思）
    s02initMean = mean(mean(s02initOutput));
    s03initMean = mean(mean(s03initOutput));
    s04initMean = mean(mean(s04initOutput));
    s05initMean = mean(mean(s05initOutput));
    s06initMean = mean(mean(s06initOutput));
    s07initMean = mean(mean(s07initOutput));

    %当层数为7时，总的初始重构为： → TotalInitResult =s01initOutput +s02initOutput+s03initOutput+s04initOutput+s05initOutput+s06initOutput+s07initOutput;
    %即：将所有层的初始重构（1个初始重构+6个初始残差）加起来，得出一个总的初始重构图像↓
    %因此，当层数为1、2、3、4、5、6、7时，初始重构为：↓
    s01Init          = net.vars(net.getVarIndex('s01combine')).value;
    s02TotalInit = net.vars(net.getVarIndex('s02combineAdds01')).value;
    s03TotalInit = net.vars(net.getVarIndex('s03combineAdds02')).value;
    s04TotalInit = net.vars(net.getVarIndex('s04combineAdds03')).value;
    s05TotalInit = net.vars(net.getVarIndex('s05combineAdds04')).value;
    s06TotalInit = net.vars(net.getVarIndex('s06combineAdds05')).value;
    s07TotalInit  = net.vars(net.getVarIndex('s07combineAdds06')).value;

    
    %各层的TotalInit经过各自层的深度重构网络后的各自残差输出（因为深度重构用的是残差网络嘛）如下：⤵
    s01FinalResidualOut = net.vars(net.getVarIndex('s01prediction')).value;    %(deep reconstruction prediction);
    s02FinalResidualOut = net.vars(net.getVarIndex('s02prediction')).value;
    s03FinalResidualOut = net.vars(net.getVarIndex('s03prediction')).value;
    s04FinalResidualOut = net.vars(net.getVarIndex('s04prediction')).value;
    s05FinalResidualOut = net.vars(net.getVarIndex('s05prediction')).value;
    s06FinalResidualOut = net.vars(net.getVarIndex('s06prediction')).value;
    s07FinalResidualOut = net.vars(net.getVarIndex('s07prediction')).value;
    
    
    %各层的TotalInit经过各自层的深度重构网络后的各自最终输出如下：↓
    s01FinalOut = net.vars(net.getVarIndex('s01dr_pred')).value;    %(deep reconstruction prediction);
    s02FinalOut = net.vars(net.getVarIndex('s02dr_pred')).value;
    s03FinalOut = net.vars(net.getVarIndex('s03dr_pred')).value;
    s04FinalOut = net.vars(net.getVarIndex('s04dr_pred')).value;
    s05FinalOut = net.vars(net.getVarIndex('s05dr_pred')).value;
    s06FinalOut = net.vars(net.getVarIndex('s06dr_pred')).value;
    s07FinalOut = net.vars(net.getVarIndex('s07dr_pred')).value;
    
switch Rate
    case 0.01
        initRec =  s01Init;
        deepResidual = s01FinalResidualOut;
        output = s01FinalOut;
    case 0.05
        initRec = s02TotalInit; 
        initResidual = s02initOutput;
        deepResidual = s02FinalResidualOut;
        output = s02FinalOut;
    case 0.1
        initRec = s03TotalInit; 
        initResidual = s03initOutput;
        deepResidual = s03FinalResidualOut;
        output = s03FinalOut;
    case 0.2
        initRec = s04TotalInit; 
        initResidual = s04initOutput;
        deepResidual = s04FinalResidualOut;
        output = s04FinalOut;
    case 0.3
        initRec = s05TotalInit; 
        initResidual = s05initOutput;
        deepResidual = s05FinalResidualOut;
        output = s05FinalOut;
    case 0.4
        initRec = s06TotalInit; 
        initResidual = s06initOutput;
        deepResidual = s06FinalResidualOut;
        output = s06FinalOut;
    case 0.5
        initRec = s07TotalInit; 
        initResidual = s07initOutput;
        deepResidual = s07FinalResidualOut;
        output = s07FinalOut;
end
end

%选择基础采样率：
% baseInitRec = s01initOutput;
% TotalInitRec = baseInitRec + initResidual;
% PSNR_s01Final = psnr(s01FinalOut,image);  %第一层初始重构的PSNR；
% PSNR_initRec = psnr(TotalInitRec,image);   %第一层初始重构+第1+2+...N层残差的PSNR；
% PSNR_output = psnr(output,image); 




if showResult
    figure;
    imshow(im2uint8(output)) ;     %sr = 0.5 finalout
    figure;
    imshow(im2uint8(s01FinalOut)); %sr = 0.01 final out
    figure;
    imshow(im2uint8(TotalInitRec));  %sr = 0.01finalout + sr = 0.5residual;
    %drawnow;
end
fprintf('当前块的采样率为：%g\n',Rate);
format long g;
save('SampleRate.txt','Rate','-append','-ascii');


%保存图像
%imwrite(s07initOutput,'./RecResult/Submarine_s07initResidual.jpg'); 



toc
end