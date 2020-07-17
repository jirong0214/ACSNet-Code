%%
% This is an implement of strategy 2(add Sample strategy );
%%
clear;
t1 = clock;
addpath('BigData');
OriginImage = imread('datasets/Test/Fish.jfif');
OriginImage = rgb2gray(OriginImage);
delete('SampleRate.txt');
blockSize = 64;

%搞一个基准的初次采样(自己选择0.01/0.05/0.1等)，并初始重构；
baseInitRec = ACS_Net_initRec(OriginImage,0.05); %0.05
subplot(1,4,1);
imshow(baseInitRec);
title('0.05采样率的基准初始重构');

%计算采样率数组，分配采样率；
fun = @(block_struct)ComputeEntropy(block_struct.data); 
Etpy  = blockproc(OriginImage,[blockSize,blockSize],fun);%熵矩阵；
E1d = Etpy(:);%获得了一个N行1列的熵向量；
E03 = roundn(E1d,-3);%保留3位小数
global Strategy2EArray;
Strategy2EArray = sort(E03);%数组排序

%分块分不同采样率补充采样；
fun = @(block_struct) ACS_Net_initResidual(block_struct.data,0.05); %基准采样率0.05；
initEnResidual  = blockproc(OriginImage,[blockSize blockSize],fun);
filename = 'SampleRate.txt';
SRs=textread(filename,'%f');
AvgSampleRate = mean(SRs);


subplot(1,4,2);
imshow(initEnResidual);
title('补充采样的初始重构残差');
OriginImage = size2same(initEnResidual,OriginImage);  %把原图裁剪为和结果图一样大的；
fprintf('Average Sample Rate = %g\n',AvgSampleRate);

TotalinitRec = initEnResidual + baseInitRec; %基准初始重构+补充采样的残差=恢复的初始重构
subplot(1,4,3);
imshow(TotalinitRec);
title('补充采样后的初始重构')

if size(OriginImage,3)==3
    OriginImage = rgb2ycbcr(OriginImage);
    OriginImage = im2double(OriginImage(:, :, 1));  %输入图像转为灰度；
else
    OriginImage =im2double(OriginImage);
end
OriginImEntropy = entropy(OriginImage);
OriginImage = size2same(TotalinitRec,OriginImage);  %把原图裁剪为和结果图一样大的；
PSNR_MultiScaleAdd_Init = psnr(double(TotalinitRec),OriginImage);
SSIM_MultiScaleAdd_Init = ssim(double(TotalinitRec),OriginImage);
subplot(1,4,4);
imshow(OriginImage);
title('原图');
t2 = clock;
runtime = etime(t2,t1);
% 
% imwrite(baseInitRec,'/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet-Code/RecResult/策略2/FIsh/baseInitRec0.05.jpg'); 
% imwrite(initEnResidual,'/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet-Code/RecResult/策略2/FIsh/initEnResidual.jpg'); 
% imwrite(TotalinitRec,'/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet-Code/RecResult/策略2/FIsh/TotalinitRec.jpg'); 
% 
