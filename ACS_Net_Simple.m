% clear all; close all;
function [output,Rate] = ACS_Net_Function(image)
    %addpath('D:\下载内容\SCSNet-master\SCSNet-Code\data\utilities');
    %run('D:\Matlab\MatConvNet\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;
    tic

    showResult  = 0; % need show result when test?
    useGPU      = 0;  
    SaveImage = 0;
    
    %switch output layer:
    Rate = 0; %采样率初始化为0；
    
    ImEntropy =  entropy(image);  %%根据图像的熵来switch采样率；
    if ImEntropy < 3.5
        Rate = 0.01;   
    elseif  ImEntropy <5
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
    
    
    %netfolder = '.\model'; %pretrained model
    netfolder = 'TrainedModel';% tianjirong's trained model folder；
    res01 = zeros(200,2);
    res02 = zeros(200,2);
    res03 = zeros(200,2);
    res04 = zeros(200,2);
    res05 = zeros(200,2);
    res06 = zeros(200,2);
    res07 = zeros(200,2);

    %netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained；  
    netpaths = dir(fullfile(netfolder,['net-epoch-50.mat']));  % switch net path
    net = load(fullfile(netfolder,netpaths.name)); 
    net = dagnn.DagNN.loadobj(net.net);

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
        label = image;
        net.conserveMemory = true;
        net.eval({'input',input});
        %Rate = 0.1; %强制赋值0.1
        if useGPU
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
            switch Rate
                    case 0.01
                        output = net.vars(net.getVarIndex('s01dr_pred')).value;
                    case 0.05
                        output = net.vars(net.getVarIndex('s02dr_pred')).value;
                    case 0.1
                        output = net.vars(net.getVarIndex('s03dr_pred')).value;
                    case 0.2
                        output = net.vars(net.getVarIndex('s04dr_pred')).value;
                    case 0.3
                        output = net.vars(net.getVarIndex('s05dr_pred')).value;
                    case 0.4
                        output = net.vars(net.getVarIndex('s06dr_pred')).value;
                    case 0.5
                        output = net.vars(net.getVarIndex('s07dr_pred')).value;
            end
        end
    toc
            if showResult
                imshow(im2uint8(output))  
                drawnow;
            end
            if SaveImage
                 imwrite(output,strcat(save_dir,['m',num2str(i),'_baby_GT.bmp']));  
            end
            fprintf('当前块的采样率为：%g\n',Rate);
            format long g;
            save('SampleRate.txt','Rate','-append','-ascii');
end