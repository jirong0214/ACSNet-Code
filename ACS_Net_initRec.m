function initRec = ACS_Net_initRec(image)
tic
%image = imread('./datasets/Test/Submarine.jfif');

%%ѡ��ѵ����ģ��·����
%netfolder = '.\model'; %pretrained model��
addpath('BigData');
netfolder = 'TrainedModel';% tianjirong's trained model folder��
%netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained��  
netpaths = dir(fullfile(netfolder,['net-epoch-50.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

%ѡ���Ƿ�ʹ��GPU��
useGPU      = 0;  
if useGPU
net.move('gpu');
end
%ѡ���Ƿ�Ҫ��ʾ�ͱ���ͼ���ع������
showResult  = 0; 
SaveImage = 0;
Rate = 0.01;




%ͼ��תΪ�����ȻҶ�ͼ��
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


%ǿ�Ƹ�ֵ�����ʶ����Ǹ���ͼ���������䣡
%Rate = 0.5; 

if useGPU
    %%ʹ��GPU������º�CPUͬ��ǰ��Ӹ�gather���ɣ��Ҿ��Ȳ����ˣ�
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
    %%����ĳ�ʼ�ع��������
    s01initOutput = net.vars(net.getVarIndex('s01combine')).value;%  ��1��ĳ�ʼ�ع�ͼ��
    s02initOutput = net.vars(net.getVarIndex('s02combine')).value;% ��2��ĳ�ʼ�ع��в
    s03initOutput = net.vars(net.getVarIndex('s03combine')).value;%��3��ĳ�ʼ�ع��в�;
    s04initOutput = net.vars(net.getVarIndex('s04combine')).value;%��4��ĳ�ʼ�ع��в�
    s05initOutput = net.vars(net.getVarIndex('s05combine')).value;%��5��ĳ�ʼ�ع��в�
    s06initOutput = net.vars(net.getVarIndex('s06combine')).value;%��6��ĳ�ʼ�ع��в�
    s07initOutput = net.vars(net.getVarIndex('s07combine')).value;%��7��ĳ�ʼ�ع��в�
    PSNR_init01 = psnr(s01initOutput,image); 
    SSIM_init01 = ssim(s01initOutput,image);
    
    
    %�����ʼ�в������ƽ��ֵ��������һ�²в�Ĵ�С,û�����˼��
    s02initMean = mean(mean(s02initOutput));
    s03initMean = mean(mean(s03initOutput));
    s04initMean = mean(mean(s04initOutput));
    s05initMean = mean(mean(s05initOutput));
    s06initMean = mean(mean(s06initOutput));
    s07initMean = mean(mean(s07initOutput));

    %������Ϊ7ʱ���ܵĳ�ʼ�ع�Ϊ�� �� TotalInitResult =s01initOutput +s02initOutput+s03initOutput+s04initOutput+s05initOutput+s06initOutput+s07initOutput;
    %���������в�ĳ�ʼ�ع���1����ʼ�ع�+6����ʼ�в���������ó�һ���ܵĳ�ʼ�ع�ͼ���
    %��ˣ�������Ϊ1��2��3��4��5��6��7ʱ����ʼ�ع�Ϊ����
    s01Init          = net.vars(net.getVarIndex('s01combine')).value;
    s02TotalInit = net.vars(net.getVarIndex('s02combineAdds01')).value;
    s03TotalInit = net.vars(net.getVarIndex('s03combineAdds02')).value;
    s04TotalInit = net.vars(net.getVarIndex('s04combineAdds03')).value;
    s05TotalInit = net.vars(net.getVarIndex('s05combineAdds04')).value;
    s06TotalInit = net.vars(net.getVarIndex('s06combineAdds05')).value;
    s07TotalInit  = net.vars(net.getVarIndex('s07combineAdds06')).value;

    
    %�����TotalInit�������Բ������ع������ĸ��Բв��������Ϊ����ع��õ��ǲв���������£��7�3
    s01FinalResidualOut = net.vars(net.getVarIndex('s01prediction')).value;    %(deep reconstruction prediction);
    s02FinalResidualOut = net.vars(net.getVarIndex('s02prediction')).value;
    s03FinalResidualOut = net.vars(net.getVarIndex('s03prediction')).value;
    s04FinalResidualOut = net.vars(net.getVarIndex('s04prediction')).value;
    s05FinalResidualOut = net.vars(net.getVarIndex('s05prediction')).value;
    s06FinalResidualOut = net.vars(net.getVarIndex('s06prediction')).value;
    s07FinalResidualOut = net.vars(net.getVarIndex('s07prediction')).value;
    
    
    %�����TotalInit�������Բ������ع������ĸ�������������£���
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

%ѡ����������ʣ�
% baseInitRec = s01initOutput;
% TotalInitRec = baseInitRec + initResidual;
% PSNR_s01Final = psnr(s01FinalOut,image);  %��һ���ʼ�ع���PSNR��
% PSNR_initRec = psnr(TotalInitRec,image);   %��һ���ʼ�ع�+��1+2+...N��в��PSNR��
% PSNR_output = psnr(output,image); 



imshow(initRec);
if showResult
    figure;
    imshow(im2uint8(output)) ;     %sr = 0.5 finalout
    figure;
    imshow(im2uint8(s01FinalOut)); %sr = 0.01 final out
    figure;
    imshow(im2uint8(TotalInitRec));  %sr = 0.01finalout + sr = 0.5residual;
    %drawnow;
end
fprintf('��ǰ��Ĳ�����Ϊ��%g\n',Rate);
format long g;
save('SampleRate.txt','Rate','-append','-ascii');


%����ͼ��
%imwrite(s07initOutput,'./RecResult/Submarine_s07initResidual.jpg'); 



toc
end