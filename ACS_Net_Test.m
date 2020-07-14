clear;
t1 = clock;
%OriginImage = imread('BigData/datasets/Test/Fish.jfif');
OriginImage = imread('BigData/datasets/Test/Man.jpg');
%OriginImage = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/ACSNet-Code/RecResult/策略1/Baby/baby_GT.bmp');
%OriginImage = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/ACSNet-Code/RecResult/策略1/butterfly/butterfly_GT.bmp');
if size(OriginImage,3)==3
    OriginImage = rgb2ycbcr(OriginImage);
    OriginImage = im2double(OriginImage(:, :, 1));  %输入图像转为灰度；
else
    OriginImage =im2double(OriginImage);
end
OriginImEntropy = entropy(OriginImage);
delete('SampleRate.txt');
PSNR = 0;
blockSize =64;
%switch work mode;
fun = @(block_struct)ComputeEntropy(block_struct.data); 
Etpy  = blockproc(OriginImage,[blockSize,blockSize],fun);%熵矩阵；
%Etpy  = Etpy1(1:size(Etpy1,1)-1,1:size(Etpy1,2)-1);
E1d = Etpy(:);%获得了一个N行1列的熵向量；
E03 = roundn(E1d,-3);%保留3位小数
global EArray;
EArray = sort(E03);%数组排序
global mode;
mode = 2; % 1为分块处理，采样率根据图像熵分配；2为整幅图像全部一块处理；3.对比测试；
global ConstRate;%设置固定的采样率；
ConstRate = 0.2;
switch mode 
    case 1  %分为bSize*bSize的块处理。
        fun = @(block_struct) ACS_Net_finalOutput(block_struct.data); 
        result  = blockproc(OriginImage,[blockSize blockSize],fun);
        %imwrite(result,'./Test/results/CurryBlocked_block64_SR_range0.01-0.5.bmp');
        filename = 'SampleRate.txt';
        SRs=textread(filename,'%f');
        AvgSampleRate = mean(SRs);
        imshow(result);
        result = double(result);
        OriginImage = size2same(result,OriginImage);  %把原图裁剪为和结果图一样大的；
        PSNR = psnr(result,OriginImage); 
        SSIM = ssim(result,OriginImage); 
        fprintf('不同块分配不同采样率时，重建的PSNR =%g\n',PSNR);
        fprintf('不同块分配不同采样率时，重建的SSIM  =%g\n',SSIM);
        fprintf('平均采样率 = %g\n',AvgSampleRate);
        %imwrite(result,'/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/ACSNet-Code/RecResult/Fish/策略1/Fish_avgSR0093.jpg'); 

    case 2   %整幅图像处理。
        result= ACS_Net_finalOutput(OriginImage);  
        imshow(result);
        result = double(result);
        OriginImage = size2same(result,OriginImage);  %把原图裁剪为和结果图一样大的；
        SSIM = ssim(result,OriginImage); 
        PSNR = psnr(result,OriginImage); 
        fprintf('单幅图像直接按同一个采样率采样时，重建的PSNR =%g\n',PSNR);
        fprintf('单幅图像直接按同一个采样率采样时，重建的SSIM =%g\n',SSIM);
        
    case 3    %整幅图像处理 +分为64*64的块处理。
        result_full = ACS_Net_Function(OriginImage); 
        fun = @(block_struct) ACS_Net_Function(block_struct.data);  
        result_blocked = blockproc(OriginImage,[64,64],fun);
        
        result_full = double(result_full);
        result_blocked = double(result_blocked);
        
        OriginImage = size2same(result,OriginImage);  %把原图裁剪为和结果图一样大的；
        PSNR_full = psnr(result_full,OriginImage); 
        PSNR_blocked = psnr(result_blocked,OriginImage); 
        
        imshow(cat(2,im2uint8(OriginImage),im2uint8(result_full),im2uint8(result_blocked)));
        fprintf('%g',PSNR_full,'%g',PSNR_blocked);
end

t2 = clock;
runtime = etime(t2,t1);
E = ComputeEntropy(OriginImage);
fprintf("所有块的平均熵大小：%d \n",mean(EArray(:)));
fprintf("熵数组为：%d",EArray);
