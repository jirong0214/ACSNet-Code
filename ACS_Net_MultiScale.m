% This is an implement of strategy 2(add Sample strategy );
clear
t1 = clock;

imageNum = 499;%Choose image nums;
image = imread(['../BigData/datasets/reference-890/',num2str(imageNum),'_img_.png']);
if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));  %输入图像转为灰度；
else
    image = im2double(image);
end
image =modcrop(image,32); % convert image -> can be mod by 32;


baseInitRec = ACS_Net_initRec(image,0.05); %base Sampling by SR of 0.05;


delete('SampleRate.txt');
blockSize = 64;
fun = @(block_struct)ComputeEntropy(block_struct.data); %Compute Entropies and distribute SRs;
Etpy  = blockproc(image,[blockSize,blockSize],fun);%Entropy matrix
E1d = Etpy(:);%获得了一个N行1列的熵向量；
E03 = roundn(E1d,-3);%保留3位小数
global Strategy2EArray;
Strategy2EArray = sort(E03);

%分块分不同采样率补充采样,获得残差；
fun = @(block_struct) MutiScaleResidual(block_struct.data); 
initEnResidual  = blockproc(image,[blockSize blockSize],fun);
filename = 'SampleRate.txt';
SRs=textread(filename,'%f');
AvgSampleRate = mean(SRs);


TotalinitRec = initEnResidual + baseInitRec; %基准初始重构+补充采样的残差=恢复的初始重构


[psnr_base,ssim_base]        = Cal_PSNRSSIM(im2uint8(image) ,im2uint8(baseInitRec),0,0);
[psnr_baseAddresidual,ssim_baseAddresidual]  = Cal_PSNRSSIM(im2uint8(image) ,im2uint8(TotalinitRec),0,0);

%show result
subplot(1,4,1);
imshow(baseInitRec);
title('0.05采样率的基准初始重构');
subplot(1,4,2);
imshow(initEnResidual);
title('补充采样的初始重构残差');
subplot(1,4,3);
imshow(TotalinitRec);
title('补充采样后的初始重构')
subplot(1,4,4);
imshow(image);
title('原图');
fprintf('Average Sample Rate = %g\n',AvgSampleRate);
t2 = clock;
runtime = etime(t2,t1);

imwrite(TotalinitRec,['../BigData/datasets/reference-890_InitRec/',num2str(imageNum),'_img_.png']); 
